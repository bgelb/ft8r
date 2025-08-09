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

### Performance tuning and profiling

This project aims to decode a 15s FT8 cycle in roughly O(1s) wall time on a modern CPU. We added a combination of algorithmic optimizations, instrumentation, and environment‑tunable knobs to balance speed and robustness.

#### Instrumentation

- `utils/prof.PROFILER` provides low‑overhead section timing when enabled via `FT8R_PROFILE=1`.
- A profiling helper `scripts/profile_decode.py` measures wall time, optionally runs `cProfile`, and can export section timing JSON.
- An opt‑in performance test `tests/test_performance.py` is available when `FT8R_PERF=1`.

Suggested usage:

```
PYTHONPATH=. FT8R_PROFILE=1 python scripts/profile_decode.py websdr_test6 --json profile.json
```

#### Implementation optimizations

- Cache the 16s full‑period FFT once per decode and reuse for baseband slicing.
- Precompute the fixed edge taper window for FFT slices.
- Cache zero‑offset tone bases per `(sample_rate, sym_len)` and apply per‑frequency phase shifts for fine frequency sync.
- Attempt both refined and legacy alignment paths (see below). A fast CRC short‑circuit on hard decisions avoids expensive LDPC when possible.

All changes preserve numerical results; any behavior differences can be gated via environment variables described next.

#### Tuning knobs (environment variables)

- `FT8R_MAX_CANDIDATES` (default: `1500`; `0` disables capping)
  - Controls how many top candidates (time/frequency peaks) are fully processed. Trade‑off: fewer candidates reduce total alignment/LDPC work (faster), but could miss marginal signals. Default chosen with headroom: across bundled samples, candidate counts are approximately p95≈1172, p99≈1244, max≈1260; we set 1500 to exceed the observed maximum.

- `FT8R_DISABLE_LEGACY` (default: `0`)
  - When `1`, skips the legacy (integer‑grid) alignment path. Trade‑off: less robust on some marginal signals but faster. Keep off for maximum coverage; enable for lower latency scenarios.

- `FT8R_RUN_BOTH` (default: `1`)
  - When `1`, runs both refined and legacy alignment for each candidate (original behavior). When `0`, run legacy only if refined fails to decode. Trade‑off: running both increases chances that at least one alignment decodes the signal at the cost of extra compute. The default favors robustness.

- `FT8R_MIN_LLR_AVG` (default: `0.0` = disabled)
  - If set to a positive float, performs an early reject when average |LLR| for a candidate is below this threshold, skipping LDPC. Trade‑off: can significantly reduce LDPC workload but risks dropping very weak decodes if set too high. Recommended only for latency‑critical deployments after tuning on your signal set.

Additional flags used in CI/testing:

- `FT8R_PERF`, `FT8R_PERF_STEM`, `FT8R_PERF_REPEATS`, `FT8R_TARGET_S`, `FT8R_PERF_ALLOW_FAIL` control the opt‑in performance test.

#### How defaults were evaluated

- We measured candidate counts across all bundled `ft8_lib-2.0` samples using `scripts/evaluate_candidate_caps.py` and saved results in `candidate_cap_eval.json`.
  - Distribution: p95≈1172, p99≈1244, max≈1260.
  - Default cap set to 1500 to exceed p99 and the observed max, providing headroom for unseen inputs while bounding worst‑case runtime.
- We profiled with `scripts/profile_decode.py` and found runtime dominated by LDPC and candidate search correlation on busy samples; the implemented caching and reuse reduced overhead meaningfully.
- Robustness: we preserved the original behavior of running both refined and legacy alignment by default (`FT8R_RUN_BOTH=1`) to avoid decode regressions while still allowing speed‑oriented configurations.

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


