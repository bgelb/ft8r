### FT8 decoder architecture

This document describes the current design of the FT8 decoder in this repository: the processing pipeline, key components, and design choices (including parabolic peak interpolation for fine synchronization).

### Overview

The decoder processes one 15‑second FT8 cycle of mono PCM audio and returns all successfully decoded messages. The pipeline is:

1) Candidate search (time/frequency grid)
2) Narrow‑band baseband extraction and decimation
3) Fine time and frequency synchronization
4) Soft demodulation to log‑likelihood ratios (LLRs)
5) LDPC(174,91) decoding
6) Message bit unpacking and text decode

### Processing pipeline

#### Candidate search (coarse sync)

- File: `search.py`
- Key functions:
  - `candidate_score_map(samples_in, max_freq_bin, max_dt_in_symbols)`
  - `peak_candidates(scores, dts, freqs, threshold)`
  - `find_candidates(samples_in, max_freq_bin, max_dt_in_symbols, threshold)`

The input audio is evaluated over a time/frequency grid using short FFTs at an oversampling ratio in time and frequency. A Costas‑sequence kernel identifies likely FT8 starts via a Costas power ratio (active bins vs. unused Costas bins), and local maxima above `threshold` are returned as `(score, dt, base_freq)` candidates.

#### Narrow‑band baseband extraction

- File: `demod.py`
- Key function: `downsample_to_baseband(samples_in, freq)`

A narrow FFT “slice” centered on the candidate base frequency is extracted, tapered at the edges, shifted to DC, and converted back to time domain at a low baseband rate (`BASEBAND_RATE_HZ`). The slice covers all 8 FT8 tones plus transition band (`SLICE_SPAN_TONES`).

#### Fine synchronization

- File: `demod.py`
- Key functions:
  - `fine_time_sync(samples, dt, search)`
  - `fine_freq_sync(samples, dt, search_hz, step_hz)`
  - `fine_sync_candidate(samples_in, freq, dt)`
  - Additional methods: `_fine_time_sync_integer`, `_fine_freq_sync_maxbin`, `_fine_sync_candidate_legacy`

Given a coarse `(dt, freq)`, the decoder refines the start time and frequency by maximizing Costas energy around the sync positions. Two complementary methods are used:

- Quadratic‑refined method: perform parabolic (quadratic) interpolation over the discrete energy samples to obtain sub‑sample timing and sub‑bin frequency offsets. This uses three‑point peak fitting around the strongest integer offset/bin.
- Integer‑grid method: use the best integer sample/bin without fractional refinement.

The decoder attempts both methods for each candidate and accepts any valid decodes produced by either alignment, providing robustness across a range of signal conditions.

Parabolic refinement uses the standard three‑point vertex estimate (clamped to ±0.5 for stability):

```
frac = 0.5 * (y_{-1} - y_{+1}) / (y_{-1} - 2*y_0 + y_{+1})
```

#### Soft demodulation

- File: `demod.py`
- Key function: `soft_demod(samples_in)`

The synchronized 79 symbols (first 7, middle 7, and last 7 are Costas) are arranged in a symbol matrix. Tone responses are computed via matched complex exponentials. Costas symbols are removed and the remaining per‑symbol tone magnitudes are normalized to probabilities. Gray‑coded bit LLRs are computed by aggregating probabilities across tones that map to 0/1 for each bit position.

#### LDPC decoding

- File: `demod.py`
- Key function: `ldpc_decode(llrs)`

LLRs are converted into per‑bit error probabilities (with reliability scaling) and fed to a BP+OSD decoder (`ldpc.BpOsdDecoder`) using the `LDPC_174_91_H` parity matrix. The corrected 174 bits (77 payload + 14 CRC + 83 parity) are returned.

#### Full‑period decode orchestration

- File: `demod.py`
- Key function: `decode_full_period(samples_in, threshold)`

Calls the candidate search, applies both fine synchronization methods, demodulates, LDPC‑decodes, and text‑decodes each valid message. Each successful decode returns a dictionary with `message`, `score`, `freq`, and `dt`.

### Design choices

- **Parabolic vs. linear or higher‑order fits**:
  - Local second‑order model: near a well‑behaved maximum, correlation/energy curves are well approximated by a quadratic Taylor expansion. Quadratic interpolation captures curvature; linear interpolation cannot represent a peak and thus gives inferior localization accuracy.
  - Efficiency vs. accuracy: quadratic interpolation uses only the peak and two neighbors, adding minimal compute while yielding sub‑grid estimates whose error decreases rapidly with SNR.
  - Robustness: higher‑order (e.g., cubic) fits require more points and are more sensitive to noise/model mismatch. Quadratic is a strong balance for small windows.

- **Dual alignment methods**:
  - Using both quadratic‑refined and integer‑grid alignment improves resilience. Some marginal signals favor the integer‑grid solution; others benefit from sub‑sample refinement. Running both avoids mode‑specific failures.

### References

- Julius O. Smith III, Spectral Audio Signal Processing, “Quadratic Interpolation of Spectral Peaks.” Stanford CCRMA. [Quadratic Interpolation of Spectral Peaks](https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html)
- R. Quinn, “Estimation of frequency by interpolation using Fourier coefficients,” IEEE Transactions on Signal Processing, 1994.
- M. I. Smith and S. F. M. Smith, “Image registration with sub‑pixel accuracy using correlation,” IEE Proceedings, 1997.


