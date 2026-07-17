"""
Decomposition Visualization for CVCBM and TSDP Experiments

Finds a noisy 60-step window (>= 10 spikes > 0.8) and produces separate images for each
decomposition stage. All decomposition steps are computed in-memory from the raw signal.

CVCBM:
  - Image 1: Original signal + ALL CEEMDAN IMFs (shared Y-scale, annotated with SE score)
  - Image 2: High-frequency Co-IMF + its VMD modes
  - Image 3: All output channels (VMD modes + Medium Co-IMF + Low Co-IMF)

TSDP:
  - Image 1: Original signal + SWT components (D1, D2, A2)
  - Image 2: D1 + its VMD modes
  - Image 3: All output channels (VMD modes + D2 + A2)
"""

import os
import sys
import argparse
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import warnings

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocessing.cskv.config import CFG as CSKV_CFG
from preprocessing.sv.config import CFG as SV_CFG
from preprocessing.cskv.decomposition import (
    ceemdan_decompose,
    sample_entropy,
    cluster_imfs,
    vmd_decompose as cskv_vmd,
)
from preprocessing.sv.decomposition import (
    vmd_decompose as sv_vmd,
)


def find_window_with_spikes(
    signal: np.ndarray,
    window_size: int = 60,
    spike_threshold: float = 0.8,
    min_spikes: int = 10,
) -> Optional[int]:
    n = len(signal)
    for i in range(n - window_size):
        w = signal[i : i + window_size]
        if np.sum(w > spike_threshold) >= min_spikes:
            return i
    return None


def load_service_signal(preprocess_dir: str, service_idx: int = 0) -> np.ndarray:
    p = os.path.join(preprocess_dir, "original", f"service_{service_idx:05d}.npy")
    return np.load(p).astype(np.float64)


def find_service_with_window(preprocess_dir: str, max_services: int = 2000) -> tuple:
    best = None
    for idx in range(max_services):
        p = os.path.join(preprocess_dir, "original", f"service_{idx:05d}.npy")
        if not os.path.exists(p):
            continue
        sig = np.load(p).astype(np.float64)
        ws = find_window_with_spikes(sig, min_spikes=10)
        if ws is not None:
            w = sig[ws:ws+60]
            transitions = int(np.sum(np.abs(np.diff(w > 0.8))))
            print(f"  Service {idx:05d}: window={ws}, spikes={int(np.sum(w > 0.8))}, "
                  f"transitions={transitions}, std={w.std():.3f}, max={w.max():.3f}")
            if best is None or transitions > best[0]:
                best = (transitions, idx, ws, sig)
    if best is None:
        raise RuntimeError(f"No service with >=10 spikes >0.8 found in first {max_services} services")
    _, idx, ws, sig = best
    w = sig[ws:ws+60]
    transitions = int(np.sum(np.abs(np.diff(w > 0.8))))
    print(f"  => Selected service {idx:05d} with {transitions} transitions (most toggling)")
    return idx, ws, sig


def plot_original_signal(ax, signal, title="Original Signal", window_start=0):
    w = signal[window_start : window_start + 60]
    ax.plot(range(60), w, linewidth=2, color="steelblue")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("CPU Utilization")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 59)


def plot_vmd_modes(fig, vmd_modes, input_signal, title="VMD Modes"):
    n_modes = vmd_modes.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_modes))

    gs = fig.add_gridspec(n_modes + 1, 1, height_ratios=[1] + [1] * n_modes, hspace=0.6)

    ax_sig = fig.add_subplot(gs[0])
    ax_sig.plot(range(60), input_signal, linewidth=2, color="gray")
    ax_sig.set_title("Input to VMD", fontsize=11, fontweight="bold")
    ax_sig.set_xlim(0, 59)
    ax_sig.set_ylabel("Amplitude")
    ax_sig.grid(True, alpha=0.3)

    for i in range(n_modes):
        ax_i = fig.add_subplot(gs[i + 1])
        ax_i.plot(range(60), vmd_modes[i], linewidth=2, color=colors[i])
        ax_i.set_title(f"VMD Mode {i+1}", fontsize=10, fontweight="bold", color=colors[i])
        ax_i.set_xlim(0, 59)
        ax_i.set_ylabel("Amplitude")
        ax_i.set_xlabel("Time Step")
        ax_i.grid(True, alpha=0.3)
        for spine in ax_i.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold")


def plot_channels(fig, channels, titles, colors, suptitle):
    n = len(channels)
    gs = fig.add_gridspec(n, 1, hspace=0.5)

    for i, (ch, t, c) in enumerate(zip(channels, titles, colors)):
        ax_i = fig.add_subplot(gs[i])
        ax_i.plot(range(60), ch, linewidth=2, color=c)
        ax_i.set_title(t, fontsize=9, fontweight="bold", color=c)
        ax_i.set_xlim(0, 59)
        ax_i.set_ylabel("Amp")
        ax_i.set_xlabel("Time Step")
        ax_i.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")


# ────────────────────────────────────────
# CVCBM helpers
# ────────────────────────────────────────

def ceemdan_decompose_and_plot(fig, signal, window_start=0):
    from sklearn.cluster import KMeans
    from matplotlib.patches import Patch

    window = signal[window_start : window_start + 60].astype(np.float64)
    imfs, _ = ceemdan_decompose(window, CSKV_CFG.CEEMDAN_EPSILON, CSKV_CFG.CEEMDAN_TRIALS)
    n_total = imfs.shape[0]

    if n_total == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No IMFs produced", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        ax.set_title("CEEMDAN — No IMFs")
        ax.axis("off")
        return None

    max_abs = float(np.abs(imfs).max())
    y_lim = (-max_abs * 1.1, max_abs * 1.1) if max_abs > 0 else (-1, 1)

    se_values = np.array([
        sample_entropy(imfs[i], CSKV_CFG.SE_M, CSKV_CFG.SE_R_FRAC, CSKV_CFG.SE_MAX_SAMPLES)
        for i in range(n_total)
    ])
    valid_mask = np.isfinite(se_values)
    if valid_mask.any():
        se_values[~valid_mask] = float(np.nanmedian(se_values))
    else:
        se_values[:] = 0.5

    k = min(CSKV_CFG.N_CLUSTERS, n_total)
    if k == 1:
        cluster_labels = np.zeros(n_total, dtype=int)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            raw_labels = km.fit_predict(se_values.reshape(-1, 1))
        cluster_mean_se = np.full(k, -np.inf, dtype=np.float64)
        for c in range(k):
            mask = raw_labels == c
            if mask.any():
                cluster_mean_se[c] = se_values[mask].mean()
        sorted_order = np.argsort(-cluster_mean_se)
        remap = np.empty(k, dtype=int)
        for new_idx, old_idx in enumerate(sorted_order):
            remap[old_idx] = new_idx
        cluster_labels = remap[raw_labels]

    cluster_colors = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#45B7D1"}
    cluster_names = {0: "High (most complex)", 1: "Medium", 2: "Low (simplest)"}

    gs = fig.add_gridspec(n_total + 1, 1, height_ratios=[1] + [1] * n_total, hspace=0.6)

    ax0 = fig.add_subplot(gs[0])
    plot_original_signal(
        ax0, signal,
        title=f"Original Signal (window [{window_start}:{window_start+59}])",
        window_start=window_start,
    )

    for i in range(n_total):
        ax_i = fig.add_subplot(gs[i + 1])
        se = se_values[i]
        se_str = f"SE={se:.4f}" if np.isfinite(se) else "SE=NaN"
        cid = cluster_labels[i]
        color = cluster_colors.get(cid, "#999999")
        ax_i.plot(range(60), imfs[i], linewidth=2, color=color)
        ax_i.set_title(f"IMF {i+1}  —  {se_str}", fontsize=10, fontweight="bold", color=color)
        ax_i.set_xlim(0, 59)
        ax_i.set_ylim(*y_lim)
        ax_i.set_ylabel("Amplitude")
        ax_i.set_xlabel("Time Step")
        ax_i.grid(True, alpha=0.3)
        for spine in ax_i.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(0.5)

    legend_elements = [
        Patch(facecolor=cluster_colors[c], label=cluster_names.get(c, f"Cluster {c}"))
        for c in sorted(set(cluster_labels))
        if c in cluster_colors
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    return imfs


# ────────────────────────────────────────
# TSDP helpers
# ────────────────────────────────────────

def get_swt_components(signal, window_start=0):
    import pywt
    w = signal[window_start : window_start + 60].astype(np.float64)
    coeffs = pywt.swt(w, "sym4", level=2, norm=True, trim_approx=True)
    return coeffs[0], coeffs[1], coeffs[2]


def plot_swt_figure(fig, signal, window_start=0):
    A2, D2, D1 = get_swt_components(signal, window_start)

    components = [
        ("D1 (Finest Detail — highest frequency)", D1, "#FF6B6B"),
        ("D2", D2, "#4ECDC4"),
        ("A2 (Approximation — lowest frequency)", A2, "#9B59B6"),
    ]

    gs = fig.add_gridspec(4, 1, height_ratios=[1] + [1] * 3, hspace=0.6)

    ax0 = fig.add_subplot(gs[0])
    plot_original_signal(
        ax0, signal,
        title=f"Original Signal (window [{window_start}:{window_start+59}])",
        window_start=window_start,
    )

    for i, (name, comp, color) in enumerate(components):
        ax_i = fig.add_subplot(gs[i + 1])
        ax_i.plot(range(60), comp, linewidth=2, color=color)
        ax_i.set_title(name, fontsize=10, fontweight="bold", color=color)
        ax_i.set_xlim(0, 59)
        ax_i.set_ylabel("Amplitude")
        ax_i.set_xlabel("Time Step")
        ax_i.grid(True, alpha=0.3)
        for spine in ax_i.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(0.5)

    return A2, D2, D1


# ────────────────────────────────────────
# Main
# ────────────────────────────────────────

def parse_service_arg(val: str) -> int:
    return int(val.lstrip("0") or "0")


def main():
    ap = argparse.ArgumentParser(description="Visualize decomposition for CVCBM and TSDP")
    ap.add_argument("--experiment", choices=["cskv", "sv", "all"], default="all",
                    help="Which experiment to visualize (default: all)")
    ap.add_argument("--service", type=str, default=None,
                    help="Service index as int (e.g. 42) or zero-padded (e.g. 00241). "
                         "Default: auto-select first with >=10 spikes >0.8")
    ap.add_argument("--cskv_preprocess_dir", default="/dataset/cskv_preprocess")
    ap.add_argument("--sv_preprocess_dir", default="/dataset/sv_preprocess")
    ap.add_argument("--out_dir", default=os.path.join(THIS_DIR, "decomposition_viz"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_cskv = args.experiment in ("cskv", "all")
    run_sv = args.experiment in ("sv", "all")

    if args.service is not None:
        service_idx = parse_service_arg(args.service)
        print(f"Loading service {service_idx:05d}...")
        cskv_sig = load_service_signal(args.cskv_preprocess_dir, service_idx) if run_cskv else None
        sv_sig = load_service_signal(args.sv_preprocess_dir, service_idx) if run_sv else None
        ref_sig = cskv_sig if cskv_sig is not None else sv_sig
        window_start = find_window_with_spikes(ref_sig)
        if window_start is None:
            print("WARNING: No window with 10+ spikes > 0.8. Trying min_spikes=5...")
            window_start = find_window_with_spikes(ref_sig, min_spikes=5)
        if window_start is None:
            print("WARNING: No suitable window found. Using index 0.")
            window_start = 0
    else:
        print("Auto-selecting service with suitable window...")
        if run_cskv:
            selected_idx, window_start, cskv_sig = find_service_with_window(args.cskv_preprocess_dir)
        elif run_sv:
            selected_idx, window_start, cskv_sig = find_service_with_window(args.sv_preprocess_dir)
        else:
            selected_idx, window_start, cskv_sig = find_service_with_window(args.cskv_preprocess_dir)
        service_idx = selected_idx
        sv_sig = load_service_signal(args.sv_preprocess_dir, service_idx) if run_sv else None
        if run_cskv and cskv_sig is None:
            cskv_sig = load_service_signal(args.cskv_preprocess_dir, service_idx)

    sv_start = min(window_start, len(sv_sig) - 60) if run_sv else 0
    ref_sig = cskv_sig if run_cskv else sv_sig
    w = ref_sig[window_start:window_start+60]
    print(f"Using window start: {window_start}")
    print(f"  spikes={int(np.sum(w > 0.8))}, std={w.std():.3f}, max={w.max():.3f}")

    # ── CVCBM ────────────────────────────────────────────────────────
    if run_cskv:
        print("\n" + "=" * 60)
        print("CVCBM")
        print("=" * 60)

        print("  Computing CEEMDAN + clustering + VMD in-memory...")
        window = cskv_sig[window_start : window_start + 60].astype(np.float64)
        imfs, residue = ceemdan_decompose(window, CSKV_CFG.CEEMDAN_EPSILON, CSKV_CFG.CEEMDAN_TRIALS)

        if imfs.shape[0] > 0:
            co_imfs = cluster_imfs(
                imfs, residue,
                m=CSKV_CFG.SE_M,
                r_frac=CSKV_CFG.SE_R_FRAC,
                max_se_samples=CSKV_CFG.SE_MAX_SAMPLES,
                n_clusters=CSKV_CFG.N_CLUSTERS,
            )

            high_freq_co_imf = co_imfs[0].copy()
            vmd_modes = cskv_vmd(
                high_freq_co_imf, K=CSKV_CFG.VMD_K, alpha=CSKV_CFG.VMD_ALPHA,
                tau=CSKV_CFG.VMD_TAU, DC=CSKV_CFG.VMD_DC,
                init=CSKV_CFG.VMD_INIT, tol=CSKV_CFG.VMD_TOL,
            )

            print(f"  CEEMDAN produced {imfs.shape[0]} IMFs")
            print(f"  VMD produced {vmd_modes.shape[0]} modes")

            print("  Image 1: CEEMDAN IMFs...")
            fig = plt.figure(figsize=(14, 4 * 6))
            ceemdan_decompose_and_plot(fig, cskv_sig, window_start=window_start)
            fig.suptitle(f"CVCBM — CEEMDAN Decomposition (Service {service_idx:05d})",
                         fontsize=14, fontweight="bold")
            p = os.path.join(args.out_dir, f"cskv_{service_idx:05d}_01_ceemdan.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"    Saved: {p}")
            if args.show:
                plt.show()
            plt.close(fig)

            print("  Image 2: VMD on High-Frequency Co-IMF...")
            fig = plt.figure(figsize=(14, 2.5 * (vmd_modes.shape[0] + 2)))
            plot_vmd_modes(fig, vmd_modes, high_freq_co_imf,
                title=f"CVCBM — VMD on High-Frequency Co-IMF (K={vmd_modes.shape[0]})")
            p = os.path.join(args.out_dir, f"cskv_{service_idx:05d}_02_vmd.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"    Saved: {p}")
            if args.show:
                plt.show()
            plt.close(fig)

            print("  Image 3: All output channels...")
            n_vmd = vmd_modes.shape[0]
            fig = plt.figure(figsize=(14, 2.5 * (n_vmd + 2)))
            vmd_colors = plt.cm.tab10(np.linspace(0, 1, n_vmd))
            channels = [vmd_modes[k] for k in range(n_vmd)] + [co_imfs[1], co_imfs[2]]
            titles = [f"VMD Mode {k+1} (from Co-IMF 0)" for k in range(n_vmd)] + [
                "Medium Frequency Component (Co-IMF 1)",
                "Low Frequency Component (Co-IMF 2)",
            ]
            colors = list(vmd_colors) + ["#4ECDC4", "#45B7D1"]
            plot_channels(fig, channels, titles, colors, "CVCBM — All 12 Output Channels")
            p = os.path.join(args.out_dir, f"cskv_{service_idx:05d}_03_outputs.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"    Saved: {p}")
            if args.show:
                plt.show()
            plt.close(fig)
        else:
            print("  Skipped (CEEMDAN produced no IMFs)")

    # ── TSDP ──────────────────────────────────────────────────────────
    if run_sv:
        print("\n" + "=" * 60)
        print("TSDP")
        print("=" * 60)

        print("  Computing SWT + VMD in-memory...")
        A2, D2, D1 = get_swt_components(sv_sig, window_start=sv_start)

        sv_vmd_modes = sv_vmd(
            D1, K=SV_CFG.VMD_K, alpha=SV_CFG.VMD_ALPHA,
            tau=SV_CFG.VMD_TAU, DC=SV_CFG.VMD_DC,
            init=SV_CFG.VMD_INIT, tol=SV_CFG.VMD_TOL,
        )
        print(f"  VMD on D1 produced {sv_vmd_modes.shape[0]} modes")

        print("  Image 1: SWT...")
        fig = plt.figure(figsize=(14, 4 * 4))
        plot_swt_figure(fig, sv_sig, window_start=sv_start)
        fig.suptitle(f"TSDP — SWT Decomposition (Service {service_idx:05d})",
                     fontsize=14, fontweight="bold")
        p = os.path.join(args.out_dir, f"sv_{service_idx:05d}_01_swt.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"    Saved: {p}")
        if args.show:
            plt.show()
        plt.close(fig)

        print("  Image 2: VMD on D1...")
        fig = plt.figure(figsize=(14, 2.5 * (sv_vmd_modes.shape[0] + 2)))
        plot_vmd_modes(fig, sv_vmd_modes, D1,
            title=f"SV — VMD on D1 (K={sv_vmd_modes.shape[0]})")
        p = os.path.join(args.out_dir, f"sv_{service_idx:05d}_02_vmd.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"    Saved: {p}")
        if args.show:
            plt.show()
        plt.close(fig)

        print("  Image 3: All output channels...")
        n_vmd = sv_vmd_modes.shape[0]
        fig = plt.figure(figsize=(14, 2.5 * (n_vmd + 2)))
        vmd_colors = plt.cm.tab10(np.linspace(0, 1, n_vmd))
        channels = [sv_vmd_modes[k] for k in range(n_vmd)] + [D2, A2]
        titles = [f"VMD Mode {k+1} (from D1)" for k in range(n_vmd)] + [
            "D2", "A2 (Approximation)",
        ]
        colors = list(vmd_colors) + ["#4ECDC4", "#9B59B6"]
        plot_channels(fig, channels, titles, colors, "TSDP — All 12 Output Channels")
        p = os.path.join(args.out_dir, f"sv_{service_idx:05d}_03_outputs.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"    Saved: {p}")
        if args.show:
            plt.show()
        plt.close(fig)

    print("\nDone.")
    print("Generated images:")
    for fname in sorted(os.listdir(args.out_dir)):
        if fname.endswith(".png"):
            print(f"  {os.path.join(args.out_dir, fname)}")


if __name__ == "__main__":
    main()
