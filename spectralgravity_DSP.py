import subprocess
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# HELPERS
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
    if len(x) < frame:
        v = np.sqrt(np.mean(x * x) + 1e-12)
        return np.array([v, v], dtype=np.float64)

    out = []
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
# STEREO ANALYSIS
# =========================

def stereo_analysis(audio):
    L = audio[:, 0]
    R = audio[:, 1]

    corr = np.corrcoef(L, R)[0, 1]
    corr = float(np.clip(corr, -1, 1))

    width = np.mean(np.abs(L - R)) / (np.mean(np.abs(L + R)) + 1e-12)

    return corr, float(width)


def stereo_guard(audio):
    corr, width = stereo_analysis(audio)

    corr_factor = 1.0 - max(0.0, 0.97 - corr) * 1.8
    corr_factor = np.clip(corr_factor, 0.72, 1.0)

    width_factor = 1.0 - max(0.0, width - 0.18) * 0.10
    width_factor = np.clip(width_factor, 0.85, 1.0)

    return float(np.clip(corr_factor * width_factor, 0.65, 1.0))


# =========================
# INTENSITY
# =========================

def intensity_estimate(audio):
    mono = audio.mean(axis=1)
    env = frame_rms(mono)

    dyn = np.std(env) / (np.mean(env) + 1e-12)
    crest = np.max(np.abs(mono)) / (np.mean(np.abs(mono)) + 1e-12)

    score = 0.65 * dyn + 0.35 * (crest / 10.0)
    return float(np.clip(score, 0.05, 0.75))


# =========================
# FILTERS
# =========================

def safe_sosfiltfilt(sos, x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 64:
        return x.copy()
    try:
        return sosfiltfilt(sos, x)
    except:
        return x.copy()


def low_sos(sr, cutoff=180.0, order=4):
    return butter(order, cutoff, btype="lowpass", fs=sr, output="sos")


def lowpass(x, sr, cutoff=180.0, order=4):
    return safe_sosfiltfilt(low_sos(sr, cutoff, order), x)


def split_bands(x, sr):
    lp_low = lowpass(x, sr, 180.0)
    lp_mid = lowpass(x, sr, 6000.0)

    low = lp_low
    mid = lp_mid - lp_low
    high = x - lp_mid

    return low, mid, high


# =========================
# GAIN MODEL
# =========================

def stable_gain_curve(signalL, signalR, intensity, min_gain=0.97, max_gain=1.03):
    envL = smooth(frame_rms(signalL), 9)
    envR = smooth(frame_rms(signalR), 9)

    env = 0.5 * (envL + envR)
    target = np.mean(env)

    gain = target / (env + 1e-12)
    gain = np.clip(gain, min_gain, max_gain)
    gain = smooth(gain, 17)

    gain = 1.0 + (gain - 1.0) * (0.25 + 0.45 * intensity)
    return gain


def apply_curve(signal, curve):
    curve = align(curve, len(signal))
    return signal * curve


# =========================
# INTERACTIVE DEBUG (PLOTLY)
# =========================

def save_debug_html_interactive(logs, out_base):
    html_path = out_base + ".html"

    fig = make_subplots(
        rows=3, cols=1,
        vertical_spacing=0.08,
        subplot_titles=("Envelope RMS", "Gain Curves", "Stereo Stats")
    )

    # Envelope
    fig.add_trace(go.Scatter(y=logs["envL"], name="L env"), row=1, col=1)
    fig.add_trace(go.Scatter(y=logs["envR"], name="R env"), row=1, col=1)

    # Gain curves
    fig.add_trace(go.Scatter(y=logs["gL"], name="Low gain"), row=2, col=1)
    fig.add_trace(go.Scatter(y=logs["gM"], name="Mid gain"), row=2, col=1)
    fig.add_trace(go.Scatter(y=logs["gH"], name="High gain"), row=2, col=1)

    # Stats
    fig.add_trace(
        go.Scatter(
            y=[logs["stereo_corr"]],
            mode="markers+text",
            text=[f"corr={logs['stereo_corr']:.3f}<br>width={logs['width']:.3f}<br>int={logs['intensity']:.3f}"],
            name="stats"
        ),
        row=3, col=1
    )

    fig.update_layout(
        title="SpectralGravity Interactive Debug",
        height=900,
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.write_html(html_path, include_plotlyjs="cdn")

    print(f"[interactive debug] {html_path}")


# =========================
# PROCESS
# =========================

def process(inp, out):
    audio, sr = load_audio(inp)

    intensity = intensity_estimate(audio)
    stereo_factor = stereo_guard(audio)
    intensity_eff = intensity * stereo_factor

    corr, width = stereo_analysis(audio)

    L = audio[:, 0]
    R = audio[:, 1]

    L_low, L_mid, L_high = split_bands(L, sr)
    R_low, R_mid, R_high = split_bands(R, sr)

    gL = stable_gain_curve(L_low, R_low, intensity_eff, 0.985, 1.015)
    gM = stable_gain_curve(L_mid, R_mid, intensity_eff, 0.985, 1.015)
    gH = stable_gain_curve(L_high, R_high, intensity_eff, 0.985, 1.015)

    outL = apply_curve(L_low, gL) + apply_curve(L_mid, gM) + apply_curve(L_high, gH)
    outR = apply_curve(R_low, gL) + apply_curve(R_mid, gM) + apply_curve(R_high, gH)

    stereo = np.stack([outL, outR], axis=1)
    stereo = soft_limiter_linked(stereo)

    save_audio(out, stereo, sr)

    logs = {
        "gL": gL,
        "gM": gM,
        "gH": gH,
        "envL": frame_rms(L),
        "envR": frame_rms(R),
        "stereo_corr": corr,
        "width": width,
        "intensity": intensity_eff
    }

    base = str(Path(out).with_suffix("").as_posix()) + "_debug"
    save_debug_html_interactive(logs, base)

    print(f"corr: {corr:.4f} width: {width:.4f} intensity: {intensity_eff:.4f}")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: script.py input output")
        raise SystemExit(1)

    process(sys.argv[1], sys.argv[2])
