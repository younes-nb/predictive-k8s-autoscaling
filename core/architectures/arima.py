import itertools
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np

from statsmodels.tsa.arima.model import ARIMA


DEFAULT_P_GRID = (0, 1, 2, 5)
DEFAULT_D_GRID = (0, 1)
DEFAULT_Q_GRID = (0, 1)


def _resolve_trend(trend: str, d: int) -> str:
    if trend != "auto":
        return trend
    return "t" if d > 0 else "c"


def select_order(
    series: np.ndarray,
    p_values: Sequence[int] = DEFAULT_P_GRID,
    d_values: Sequence[int] = DEFAULT_D_GRID,
    q_values: Sequence[int] = DEFAULT_Q_GRID,
    trend: str = "auto",
) -> Tuple[int, int, int]:
    best = None
    best_aic = np.inf
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                fit = ARIMA(series, order=(p, d, q), trend=_resolve_trend(trend, d)).fit(
                    method_kwargs={"disp": False}
                )
        except Exception:
            continue
        if fit.aic < best_aic:
            best_aic = fit.aic
            best = (p, d, q)
    if best is None:
        return (1, 1, 0)
    return best


class ArimaForecaster:

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        trend: str = "auto",
    ):
        self.order = order
        self.trend = trend
        self._fit_result = None
        self._train_length = 0
        self.aic: Optional[float] = None

    def fit(
        self,
        series: np.ndarray,
        order: Optional[Tuple[int, int, int]] = None,
        trend: Optional[str] = None,
    ) -> "ArimaForecaster":
        if order is None:
            order = self.order
        if order is None:
            order = select_order(np.asarray(series, dtype=float), trend=self.trend)
        if trend is None:
            trend = self.trend

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            fit_result = ARIMA(
                np.asarray(series, dtype=float),
                order=order,
                trend=_resolve_trend(trend, order[1]),
            ).fit(method_kwargs={"disp": False})

        self.order = order
        self.trend = trend
        self._fit_result = fit_result
        self._train_length = len(series)
        self.aic = float(fit_result.aic)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("ArimaForecaster.forecast() called before fit()")
        return np.asarray(self._fit_result.forecast(steps), dtype=np.float64)

    def predict(self, start: int, end: int) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("ArimaForecaster.predict() called before fit()")
        idx = self._fit_result.get_prediction(start=start, end=end)
        return np.asarray(idx.predicted_mean, dtype=np.float64)

