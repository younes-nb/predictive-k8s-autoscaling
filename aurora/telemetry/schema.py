"""Feature schema: what an observation contains, and how missing data is represented.

Three states per feature, never collapsed into one:

  * ``value`` is a real number            -> mask bit 1
  * ``value`` is ``None`` (unavailable)   -> mask bit 0, value 0.0
  * ``value`` is ``None`` + stale flag    -> mask bit 0, stale flag set

A zero and an unavailable metric are semantically different and the model must
be able to tell them apart, so the mask travels with the value.
"""

from dataclasses import dataclass, field
import math


@dataclass
class Observation:
    """One 5 s observation for one service."""

    timestamp: float                 # epoch seconds, the sample's own timestamp
    service: str
    values: dict                     # feature name -> float
    available: dict                  # feature name -> bool
    quality: dict = field(default_factory=dict)   # feature -> "ok"|"stale"|"missing"

    def value(self, name):
        return float(self.values.get(name, 0.0))

    def has(self, name):
        return bool(self.available.get(name, False))


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def make_observation(timestamp, service, raw, features):
    """Build an Observation from raw metric values.

    Non-finite / None -> unavailable (mask 0), never a silent zero.
    """
    values, available, quality = {}, {}, {}
    for name in features:
        v = raw.get(name)
        if _finite(v):
            values[name] = float(v)
            available[name] = True
            quality[name] = "ok"
        else:
            values[name] = 0.0
            available[name] = False
            quality[name] = "missing"
    return Observation(timestamp=timestamp, service=service, values=values,
                       available=available, quality=quality)


@dataclass
class Batch:
    """A model-ready window: ordered observations, aligned on a fixed grid."""

    timestamps: list                 # oldest -> newest, epoch seconds
    matrix: list                     # [T][F] floats (0 where masked)
    mask: list                       # [T][F] bool
    service: str
    complete: bool                   # False if any row is missing/stale

    @property
    def n_steps(self):
        return len(self.timestamps)

    def last_row(self):
        return self.matrix[-1], self.mask[-1]