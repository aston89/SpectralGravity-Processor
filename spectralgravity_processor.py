import subprocess
import numpy as np
from scipy.signal import butter, sosfiltfilt

# =========================
# IO
# =========================

def load_audio(path, sr=44100):
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "2",
        "-ar", str(sr),
        "-"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = p.stdout.read()
    audio = np.frombuffer(raw, dtype=np.float32)

    if audio.size % 2 != 0:
        audio = audio[:-1]

    return audio.reshape(-1, 2), sr


def save_audio(path, audio, sr=44100):
    audio = np.asarray(audio, dtype=np.float32)
    p = subprocess.Popen([
        "ffmpeg",
        "-f", "f32le",
        "-ar", str(sr),
        "-ac", "2",
        "-i", "-",
        path
    ], stdin=subprocess.PIPE)

    p.stdin.write(audio.tobytes())
    p.stdin.close()
    p.wait()


# =========================
# BASIC HELPERS
# =========================

def align(values, n):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.ones(n, dtype=np.float64)

    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, values)


def smooth(x, k=11):
    k = max(3, int(k) | 1)
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same")


def frame_rms(x, frame=4096, hop=1024):
    x = np.asarray(x, dtype=np.float64)
    out = []
    if len(x) < frame:
        out.append(np.sqrt(np.mean(x * x) + 1e-12))
        out.append(out[0])
        return np.array(out, dtype=np.float64)

    for i in range(0, len(x) - frame + 1, hop):
        f = x[i:i + frame]
        out.append(np.sqrt(np.mean(f * f) + 1e-12))
    if len(out) < 2:
        out.append(out[0])
    return np.array(out, dtype=np.float64)


def soft_limiter_linked(stereo, ceiling=0.95):
    peak = np.max(np.abs(stereo)) + 1e-12
    if peak <= ceiling:
        return stereo
    return stereo * (ceiling / peak)


# =========================
# STEREO ANALYSIS / GUARD
# =========================

def stereo_analysis(audio):
    L = audio[:, 0]
    R = audio[:, 1]

    if len(L) < 2:
        return 1.0, 0.0

    corr = np.corrcoef(L, R)[0, 1]
    corr = float(np.clip(corr, -1.0, 1.0))

    width = np.mean(np.abs(L - R)) / (np.mean(np.abs(L + R)) + 1e-12)
    width = float(width)

    return corr, width


def stereo_guard(audio):
    corr, width = stereo_analysis(audio)

    # stereo molto incoerente -> meno intervento
    corr_factor = 1.0 - max(0.0, 0.97 - corr) * 1.8
    corr_factor = np.clip(corr_factor, 0.72, 1.0)

    # width già ampia -> ancora meno intervento
    width_factor = 1.0 - max(0.0, width - 0.18) * 0.10
    width_factor = np.clip(width_factor, 0.85, 1.0)

    return float(np.clip(corr_factor * width_factor, 0.65, 1.0))


# =========================
# INTENSITY ESTIMATE
# =========================

def intensity_estimate(audio):
    mono = audio.mean(axis=1)
    env = frame_rms(mono, frame=4096, hop=1024)

    dyn = np.std(env) / (np.mean(env) + 1e-12)
    crest = np.max(np.abs(mono)) / (np.mean(np.abs(mono)) + 1e-12)

    score = 0.65 * dyn + 0.35 * (crest / 10.0)
    return float(np.clip(score, 0.05, 0.75))


# =========================
# CROSSOVERS (ZERO-PHASE, RECONSTRUCTABLE)
# =========================

def safe_sosfiltfilt(sos, x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 64:
        return x.copy()
    try:
        return sosfiltfilt(sos, x)
    except Exception:
        return x.copy()


def low_sos(sr, cutoff=180.0, order=4):
    return butter(order, cutoff, btype="lowpass", fs=sr, output="sos")


def lowpass(x, sr, cutoff=180.0, order=4):
    return safe_sosfiltfilt(low_sos(sr, cutoff=cutoff, order=order), x)


def split_bands(x, sr):
    # exact-ish reconstruction by linear decomposition
    lp_low = lowpass(x, sr, cutoff=180.0, order=4)
    lp_mid = lowpass(x, sr, cutoff=6000.0, order=4)

    low = lp_low
    mid = lp_mid - lp_low
    high = x - lp_mid

    return low, mid, high


# =========================
# STABLE GAIN CURVES
# =========================

def stable_gain_curve(signalL, signalR, intensity, min_gain=0.97, max_gain=1.03):
    envL = smooth(frame_rms(signalL, frame=4096, hop=1024), 9)
    envR = smooth(frame_rms(signalR, frame=4096, hop=1024), 9)

    env = 0.5 * (envL + envR)
    target = np.mean(env)

    gain = target / (env + 1e-12)
    gain = np.clip(gain, min_gain, max_gain)
    gain = smooth(gain, 17)

    # intensity nudges, but does not dominate
    gain = 1.0 + (gain - 1.0) * (0.25 + 0.45 * intensity)
    return gain


def apply_curve(signal, curve):
    curve = align(curve, len(signal))
    return signal * curve


# =========================
# PROCESS
# =========================

def process(inp, out):
    audio, sr = load_audio(inp)

    intensity = intensity_estimate(audio)
    stereo_factor = stereo_guard(audio)
    intensity_eff = intensity * stereo_factor

    corr, width = stereo_analysis(audio)
    print(f"intensity: {intensity_eff:.6f}  stereo_corr: {corr:.6f}  width: {width:.6f}")

    L = audio[:, 0]
    R = audio[:, 1]

    # split each channel separately, but with linked control curves
    L_low, L_mid, L_high = split_bands(L, sr)
    R_low, R_mid, R_high = split_bands(R, sr)

    # one shared curve per band, derived from both channels
    gL = stable_gain_curve(L_low, R_low, intensity_eff, min_gain=0.985, max_gain=1.015)
    gM = stable_gain_curve(L_mid, R_mid, intensity_eff, min_gain=0.985, max_gain=1.015)
    gH = stable_gain_curve(L_high, R_high, intensity_eff, min_gain=0.985, max_gain=1.015)

    # apply the same linked curve to left and right band components
    L_low_p = apply_curve(L_low, gL)
    R_low_p = apply_curve(R_low, gL)

    L_mid_p = apply_curve(L_mid, gM)
    R_mid_p = apply_curve(R_mid, gM)

    L_high_p = apply_curve(L_high, gH)
    R_high_p = apply_curve(R_high, gH)

    # reconstruct stereo without collapsing to mono
    outL = L_low_p + L_mid_p + L_high_p
    outR = R_low_p + R_mid_p + R_high_p

    stereo = np.stack([outL, outR], axis=1)

    # linked final safety ceiling
    stereo = soft_limiter_linked(stereo, ceiling=0.95)

    save_audio(out, stereo, sr)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: spectralgravity_processor.py input.file output.file")
        raise SystemExit(1)

    process(sys.argv[1], sys.argv[2])
