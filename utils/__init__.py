from dataclasses import dataclass
import wave
import struct
from typing import List

# Order of tones forming the 7-symbol Costas sync sequence.  Each value
# is the tone index (0-7) transmitted for one of the seven sync symbols.
# The arrangement ensures that every tone pair is unique, allowing the
# receiver to reliably determine time and frequency alignment.
COSTAS_SEQUENCE = [3, 1, 4, 0, 6, 5, 2]

# Spacing between adjacent FT8 tones.
TONE_SPACING_IN_HZ = 6.25

# By convention FT8 transmissions begin 0.5 seconds into the 15 second
# cycle.  ``ft8sim`` therefore generates audio starting 0.5 seconds after
# the nominal start of the cycle.  Subtract this value from detected
# start times so they line up with timestamps reported by WSJT‑X.
COSTAS_START_OFFSET_SEC = 0.5


@dataclass
class RealSamples:
    """Simple container for real-valued samples and their sampling rate."""

    samples: List[float]
    sample_rate_in_hz: int


def read_wav(path: str) -> RealSamples:
    """Load mono PCM WAV data and return a :class:`RealSamples` object."""
    with wave.open(path, "rb") as w:
        nframes = w.getnframes()
        sample_rate_in_hz = w.getframerate()
        frames = w.readframes(nframes)
        sample_width = w.getsampwidth()
        if sample_width == 2:
            fmt = f"<{nframes}h"
            max_val = 32768.0
        elif sample_width == 1:
            fmt = f"<{nframes}b"
            max_val = 128.0
        else:
            raise ValueError("Unsupported sample width")
        ints = struct.unpack(fmt, frames)
        samples = [s / max_val for s in ints]
        return RealSamples(samples=samples, sample_rate_in_hz=sample_rate_in_hz)

__all__ = [
    "RealSamples",
    "read_wav",
    "COSTAS_SEQUENCE",
    "TONE_SPACING_IN_HZ",
    "COSTAS_START_OFFSET_SEC",
]
