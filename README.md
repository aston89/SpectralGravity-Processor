# SpectralGravity Processor

SpectralGravity Processor is a lightweight perceptual audio stabilizer designed to gently rebalance spectral energy over time without sounding compressed, overprocessed or artificially mastered.

Unlike traditional compressors, exciters or dynamic EQs, SpectralGravity works through extremely small multiband gain corrections that act more like a "spectral gravity field" than a conventional audio processor.

The goal is not loudness.

The goal is stability.

---

# What It Does

The processor:

- Splits audio into Low / Mid / High bands
- Measures energy distribution over time
- Applies extremely subtle linked gain corrections
- Preserves stereo width and phase integrity
- Reduces harshness and unstable spectral spikes
- Maintains the original tonal identity of the mix

All processing is performed using:
- zero-phase filtering
- stereo-linked control curves
- reconstructable band decomposition
- ultra-low gain ranges

The result is a smoother and more coherent listening experience without the audible artifacts of aggressive mastering processors.

---

# Philosophy

Most audio processors attempt to *shape* sound.

SpectralGravity attempts to *stabilize* it.

Instead of forcing a mix toward a target tone, the processor softly nudges unstable energy back toward equilibrium using microscopic adaptive corrections.

This creates a subtle but perceptually significant effect:

- less listening fatigue
- smoother high frequencies
- more consistent tonal balance
- better spectral glue
- fewer aggressive transient spikes

without destroying dynamics or stereo imaging.

---

# Why It Sounds Different

The processor operates within extremely small gain ranges:

```python
min_gain = 0.985
max_gain = 1.015
````

This is intentional.

Large corrections quickly become audible and destructive.
SpectralGravity instead relies on continuous statistical micro-corrections distributed over time.

The effect is cumulative and psychoacoustic rather than overt.

---

# Core Features

## Stereo-Safe Processing

The processor analyzes both channels together while preserving independent channel reconstruction.

This avoids:

* stereo collapse
* center overfocusing
* mono-like narrowing
* unstable image shifts

A dedicated stereo protection stage automatically reduces intervention on highly decorrelated or naturally wide mixes.

---

## Zero-Phase Filtering

Band splitting uses zero-phase filtering to avoid phase distortion and preserve transient integrity.

This makes the processor suitable even for delicate stereo material and mastering-like workflows.

---

## Adaptive Intensity

The processing intensity automatically adapts to the source material using:

* dynamic range estimation
* crest factor analysis
* stereo coherence analysis

More unstable mixes receive slightly stronger stabilization.
Well-balanced mixes remain mostly untouched.

---

# Band Structure

The signal is separated into:

| Band | Range          |
| ---- | -------------- |
| Low  | < 180 Hz       |
| Mid  | 180 Hz – 6 kHz |
| High | > 6 kHz        |

The decomposition is reconstructable:

```text
Low + Mid + High ≈ Original Signal
```

---

# Best Use Cases

SpectralGravity performs especially well on:

* amateur or inconsistent mixes
* harsh headphone content
* overly sharp high frequencies
* unstable electronic music
* cinematic and ambient material
* live recordings
* mixes with transient imbalance
* tracks that feel fatiguing over time

It is particularly effective when a mix is "almost good" but still feels unstable or tiring.

---

# Cases Where It Helps Less

The processor may have minimal impact on:

* extremely polished mastering-grade mixes
* intentionally raw material
* highly dynamic orchestral recordings
* mixes designed around aggressive transient contrast

Because the processor is intentionally conservative, it rarely damages audio, but the perceptual effect may become very subtle on already optimized material.

---

# Technical Overview

## Processing Pipeline

1. Audio decoding via FFmpeg
2. Stereo analysis
3. Dynamic intensity estimation
4. Zero-phase multiband split
5. Shared stereo-linked gain curve generation
6. Independent channel reconstruction
7. Final linked safety limiter

---

# Requirements

* Python 3.9+
* FFmpeg
* NumPy
* SciPy

Install dependencies:

```bash
pip install numpy scipy
```

Install FFmpeg:
[https://ffmpeg.org/](https://ffmpeg.org/)

---

# Usage

```bash
python spectralgravity_processor.py input.wav output.wav
```

Example:

```bash
python spectralgravity_processor.py mix.wav mix_processed.wav
```

---

# Output Philosophy

SpectralGravity is designed so that:

* bypass comparisons may initially feel subtle
* long listening sessions reveal the real effect
* the mix feels calmer rather than "processed"

---

# Disclaimer

This is not an AI mastering tool.
It does not perform intelligent source separation, stem balancing or genre-aware mastering.

It is a perceptual stabilization processor built around statistical spectral redistribution and psychoacoustic smoothing principles.

---

# Notes

SpectralGravity was created from the idea that many mixes do not fail because of catastrophic problems but because of small unstable energy accumulations distributed across time (especially aggressive pumping).
Instead of attempting to redesign a mix, the processor simply reduces spectral turbulence.
Sometimes that is enough.
