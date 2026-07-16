"""
Successive Variational Mode Decomposition (SVMD).

Pure Python port of the MATLAB reference implementation by Nazari & Sakhaei:
  M. Nazari, S. M. Sakhaei, "Successive Variational Mode Decomposition,"
  Signal Processing, Vol. 174, September 2020.
  https://doi.org/10.1016/j.sigpro.2020.107610
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def svmd(
    signal: np.ndarray,
    max_alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-6,
    stop_criteria: int = 2,
    init_omega: int = 0,
    max_modes: int = 20,
    max_inner_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Successive Variational Mode Decomposition.

    Parameters
    ----------
    signal : 1-D array
        Input time series.
    max_alpha : float
        Maximum balancing parameter. Alpha escalates from 10 toward this.
    tau : float
        Dual-ascent time-step. 0 for noisy signals.
    tol : float
        Convergence tolerance.
    stop_criteria : int
        1 = noise, 2 = exact reconstruction, 3 = BIC, 4 = power-of-last-mode.
    init_omega : int
        0 = center freq from zero, 1 = random.
    max_modes : int
        Hard upper limit on number of modes to extract.
    max_inner_iter : int
        Max iterations per alpha value.

    Returns
    -------
    modes : (n_modes, save_T), modes_hat : (save_T, n_modes), center_freqs : (n_modes,)
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    save_T = len(signal)
    if save_T < 4:
        return signal[np.newaxis, :], signal[:, np.newaxis], np.array([0.0])

    if save_T % 2 != 0:
        signal = signal[:-1]
        save_T -= 1

    eps = np.finfo(np.float64).eps
    N = max_inner_iter
    min_alpha = 10.0

    # --- Part 1: noise estimation, mirroring, FFT ---

    sg_window = min(25, save_T if save_T % 2 == 1 else save_T - 1)
    if sg_window < 9:
        sg_window = 9
    y_smooth = savgol_filter(signal, sg_window, 8)
    signoise = signal - y_smooth

    T_half = save_T // 2
    f_mir = np.zeros(2 * save_T)
    f_mir[:T_half] = signal[T_half - 1::-1]
    f_mir[T_half:3 * T_half] = signal
    f_mir[3 * T_half:] = signal[save_T - 1:T_half - 1:-1]

    f_mir_noise = np.zeros(2 * save_T)
    f_mir_noise[:T_half] = signoise[T_half - 1::-1]
    f_mir_noise[T_half:3 * T_half] = signoise
    f_mir_noise[3 * T_half:] = signoise[save_T - 1:T_half - 1:-1]

    T = 2 * save_T

    t = np.arange(1, T + 1, dtype=np.float64) / T
    omega_freqs = t - 0.5 - 1.0 / T

    f_hat = np.fft.fftshift(np.fft.fft(f_mir))
    f_hat_onesided = f_hat.copy()
    f_hat_onesided[:T // 2] = 0.0

    f_hat_n = np.fft.fftshift(np.fft.fft(f_mir_noise))
    f_hat_n_onesided = f_hat_n.copy()
    f_hat_n_onesided[:T // 2] = 0.0
    noisepe = float(np.linalg.norm(f_hat_n_onesided, 2) ** 2)

    # --- Part 2+3: main SVMD loop ---

    Alpha = min_alpha
    SC2 = False

    h_hat_list: list[np.ndarray] = []
    u_hat_i_list: list[np.ndarray] = []
    omega_d_temp: list[float] = []
    alpha_temp: list[float] = []
    u_hat_temp_list: list[np.ndarray] = []
    polm_list: list[float] = []
    polm_temp = 0.0
    l = 0

    while not SC2:

        m_val = 0.0
        bf = 0

        while Alpha < max_alpha + 1:

            omega_L = np.zeros(N, dtype=np.float64)
            if init_omega > 0:
                rng = np.random.default_rng()
                omega_L[0] = float(np.exp(np.log(eps) + (np.log(0.5) - np.log(eps)) * rng.random()))

            udiff = tol + eps
            lambda_arr = np.zeros((N, T), dtype=np.complex128)
            u_hat_L = np.zeros((N, T), dtype=np.complex128)
            n = 0

            sum_h = (np.sum(h_hat_list, axis=0)
                     if h_hat_list else np.zeros(T))
            sum_u_hat_i = (np.sum(u_hat_i_list, axis=0)
                           if u_hat_i_list else np.zeros(T, dtype=np.complex128))

            while udiff > tol and n < N - 1:
                omega_diff = omega_freqs - omega_L[n]
                od2 = omega_diff ** 2
                od4 = od2 ** 2

                numer = (f_hat_onesided
                         + Alpha ** 2 * od4 * u_hat_L[n]
                         + lambda_arr[n] / 2.0)
                denom = (1.0
                         + Alpha ** 2 * od4 * (1.0 + 2.0 * Alpha * od2)
                         + sum_h)
                u_hat_L[n + 1] = numer / denom

                upper_abs2 = np.abs(u_hat_L[n + 1, T // 2:]) ** 2
                s = np.sum(upper_abs2)
                if s > eps:
                    omega_L[n + 1] = float(np.dot(omega_freqs[T // 2:], upper_abs2) / s)

                lambda_arr[n + 1] = (
                    lambda_arr[n]
                    + tau * (f_hat_onesided
                             - u_hat_L[n + 1]
                             - ((Alpha ** 2 * od4
                                 * (f_hat_onesided - u_hat_L[n + 1]
                                    - sum_u_hat_i + lambda_arr[n] / 2.0)
                                 - sum_u_hat_i)
                                / (1.0 + Alpha ** 2 * od4))
                             - sum_u_hat_i)
                )

                diff = u_hat_L[n + 1] - u_hat_L[n]
                energy_prev = float(np.sum(np.abs(u_hat_L[n]) ** 2))
                udiff = abs(eps + (float(np.sum(np.abs(diff) ** 2)) / energy_prev
                                   if energy_prev > eps else float(np.sum(np.abs(diff) ** 2))))
                n += 1

            # Alpha escalation
            if abs(m_val - np.log(max_alpha)) > 1:
                m_val += 1
            else:
                m_val += 0.05
                bf += 1

            if bf >= 2:
                Alpha += 1

            if Alpha <= max_alpha - 1:
                if bf == 1:
                    Alpha = max_alpha - 1
                else:
                    Alpha = np.exp(m_val)

                omega_L_prev = omega_L[n]
                temp_ud = u_hat_L[n].copy()

                udiff = tol + eps
                lambda_arr = np.zeros((N, T), dtype=np.complex128)
                u_hat_L = np.zeros((N, T), dtype=np.complex128)
                n = 0
                omega_L = np.zeros(N, dtype=np.float64)
                omega_L[0] = omega_L_prev
                u_hat_L[0] = temp_ud

                sum_h = (np.sum(h_hat_list, axis=0)
                         if h_hat_list else np.zeros(T))
                sum_u_hat_i = (np.sum(u_hat_i_list, axis=0)
                               if u_hat_i_list else np.zeros(T, dtype=np.complex128))
            else:
                break

        # --- Save mode ---
        omega_L_nz = omega_L[omega_L > 0]
        omega_val = float(omega_L_nz[-1]) if len(omega_L_nz) > 0 else float(omega_L[max(0, n)])

        u_hat_temp_list.append(u_hat_L[n].copy())
        omega_d_temp.append(omega_val)
        alpha_temp.append(Alpha)

        Alpha = min_alpha

        if init_omega > 0:
            rng = np.random.default_rng()
            for _ in range(300):
                cand = float(np.exp(np.log(eps) + (np.log(0.5) - np.log(eps)) * rng.random()))
                if all(abs(cand - od) >= 0.02 for od in omega_d_temp):
                    break

        u_hat_i_list.append(u_hat_L[n].copy())

        alpha_l = alpha_temp[-1]
        omega_c = omega_d_temp[-1]
        denom_h = (alpha_l ** 2) * (omega_freqs - omega_c) ** 4
        denom_h = np.where(np.abs(denom_h) < eps, eps, denom_h)
        h_hat_list.append(1.0 / denom_h)

        l += 1

        # --- Stopping criteria ---
        if stop_criteria == 1:
            stacked = np.sum(u_hat_i_list, axis=0)
            sigerror = float(np.linalg.norm(f_hat_onesided - stacked, 2) ** 2)
            if sigerror <= round(noisepe):
                SC2 = True

        elif stop_criteria == 2:
            sum_u = np.sum(u_hat_temp_list, axis=0)
            normind = (float(np.linalg.norm(sum_u - f_hat_onesided) ** 2)
                       / float(np.linalg.norm(f_hat_onesided) ** 2 + eps))
            if normind < 0.005:
                SC2 = True

        elif stop_criteria == 3:
            stacked = np.sum(u_hat_i_list, axis=0)
            sigerror = float(np.linalg.norm(f_hat_onesided - stacked, 2) ** 2)
            bic_l = 2 * T * np.log(sigerror + eps) + (3 * l) * np.log(2 * T)
            if l > 1 and bic_l > polm_list[-1]:
                SC2 = True
            polm_list.append(bic_l)

        else:
            # Stop when power of last mode is negligible relative to first
            H = (4.0 * alpha_temp[-1]
                 / (1.0 + 2.0 * alpha_temp[-1] * (omega_freqs - omega_d_temp[-1]) ** 2))
            polm_l = float(np.abs(np.dot(H, np.abs(u_hat_i_list[-1]) ** 2)))
            if l <= 1:
                polm_temp = max(polm_l, eps)
                polm_normed = 1.0 if polm_l > 0 else 0.0
            else:
                polm_normed = polm_l / polm_temp if polm_temp > 0 else 0.0

            if l > 1 and polm_normed < tol:
                SC2 = True

        if l >= max_modes:
            SC2 = True

    # --- Reconstruction ---
    L = len(omega_d_temp)
    if L == 0:
        return signal[np.newaxis, :], signal[:, np.newaxis], np.array([0.0])

    u_hat_all = np.column_stack(u_hat_temp_list)

    u_hat_full = np.zeros((T, L), dtype=np.complex128)
    upper = slice(T // 2, T)
    u_hat_full[upper, :] = u_hat_all[upper, :]
    u_hat_full[1:T // 2 + 1, :] = np.conj(u_hat_all[upper, :][::-1, :])
    u_hat_full[0, :] = np.conj(u_hat_full[-1, :])

    u = np.zeros((L, T), dtype=np.float64)
    for li in range(L):
        u[li, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat_full[:, li])))

    order = np.argsort(omega_d_temp)
    u = u[order, :]
    omega_sorted = np.array(omega_d_temp, dtype=np.float64)[order]

    u = u[:, T // 4:3 * T // 4]

    u_hat_out = np.zeros((save_T, L), dtype=np.complex128)
    for li in range(L):
        u_hat_out[:, li] = np.fft.fftshift(np.fft.fft(u[li, :]))

    return u, u_hat_out, omega_sorted
