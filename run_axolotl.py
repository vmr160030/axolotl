# standard
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import h5py
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import math
import gc
from collections import defaultdict
import time
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# custom

import axolotl_utils_ram
importlib.reload(axolotl_utils_ram)

from axolotl_utils_ram import extract_snippets_fast_ram


import lighthouse_utils
importlib.reload(lighthouse_utils)

from lighthouse_utils import (
    find_valley_and_times
)

import plot_ei_waveforms
importlib.reload(plot_ei_waveforms)
import plot_ei_waveforms as pew

import collision_utils
importlib.reload(collision_utils)


import compute_sta_from_spikes
importlib.reload(compute_sta_from_spikes)

import joint_utils
importlib.reload(joint_utils)

import retinanalysis as ra

_PARALLEL_RAW_MOD = None
_PARALLEL_CFG = None

# ============================================================
# LH + BL/TR support helpers
# ============================================================
VERBOSE_TIMING = True
def _stage_start(label, channel_tag=None):
    if VERBOSE_TIMING:
        prefix = f"[CH {channel_tag}] " if channel_tag is not None else ""
        print(f"{prefix}START {label}")
    return time.perf_counter()

def _stage_end(label, t0, timings=None, channel_tag=None):
    dt = time.perf_counter() - t0
    if timings is not None:
        timings[label] = dt
    if VERBOSE_TIMING:
        prefix = f"[CH {channel_tag}] " if channel_tag is not None else ""
        print(f"{prefix}DONE  {label} | {dt:.2f} s")
    return dt

def _timed_call(label, func, *args, channel_tag=None, timings=None, **kwargs):
    t0 = _stage_start(label, channel_tag=channel_tag)
    try:
        return func(*args, **kwargs)
    finally:
        _stage_end(label, t0, timings=timings, channel_tag=channel_tag)
        
def _flatten_masked_snips(snips_12_l_n, mask_12_l):
    snips_12_l_n = np.asarray(snips_12_l_n, dtype=np.float32)
    mask_12_l = np.asarray(mask_12_l, dtype=bool)
    assert snips_12_l_n.ndim == 3
    assert mask_12_l.shape == snips_12_l_n.shape[:2]
    X = snips_12_l_n.transpose(2, 0, 1).reshape(snips_12_l_n.shape[2], -1)
    return X[:, mask_12_l.ravel()].astype(np.float32)


def _row_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)


def _resolve_k_list(k_list, n_available):
    return [int(k) for k in k_list if int(k) >= 1 and int(k) <= int(n_available)]


def _topk_mean_curve(sorted_desc):
    sorted_desc = np.asarray(sorted_desc, dtype=np.float32)
    return np.cumsum(sorted_desc) / np.arange(1, sorted_desc.size + 1, dtype=np.float32)


def _support_metrics_from_curves(bl_curve, tr_curve, k_peak, k_bulk):
    kmax = min(bl_curve.size, tr_curve.size)
    if kmax < 1:
        raise ValueError("Need at least one neighbor on each side.")
    k_peak_use = _resolve_k_list(k_peak, kmax)
    k_bulk_use = _resolve_k_list(k_bulk, kmax)
    if len(k_peak_use) == 0 or len(k_bulk_use) == 0:
        raise ValueError(f"k lists invalid for kmax={kmax}")

    bl_peak = float(np.mean([bl_curve[k - 1] for k in k_peak_use]))
    tr_peak = float(np.mean([tr_curve[k - 1] for k in k_peak_use]))
    bl_bulk = float(np.mean([bl_curve[k - 1] for k in k_bulk_use]))
    tr_bulk = float(np.mean([tr_curve[k - 1] for k in k_bulk_use]))
    d_peak = bl_peak - tr_peak
    d_bulk = bl_bulk - tr_bulk

    return dict(
        kmax=int(kmax),
        k_peak_used=k_peak_use,
        k_bulk_used=k_bulk_use,
        BL_peak=bl_peak,
        TR_peak=tr_peak,
        BL_bulk=bl_bulk,
        TR_bulk=tr_bulk,
        D_peak=d_peak,
        D_bulk=d_bulk,
    )


def _assign_support_label(metrics, min_bl_bulk, diag_eps):
    bl_bulk = float(metrics["BL_bulk"])
    d_bulk = float(metrics["D_bulk"])  # BL_bulk - TR_bulk

    if abs(d_bulk) <= float(diag_eps):
        return "uncertain_boundary"

    if d_bulk > float(diag_eps):
        if bl_bulk >= float(min_bl_bulk):
            return "LH"
        return "uncertain_lowBL"

    return "soup"


def _compute_one_spike_metrics(v, X_bl_n, X_tr_n, side, idx, k_peak, k_bulk):
    cos_bl = X_bl_n @ v
    cos_tr = X_tr_n @ v

    if side == "BL":
        cos_bl = cos_bl.copy()
        if 0 <= int(idx) < cos_bl.size:
            cos_bl[int(idx)] = np.nan
    elif side == "TR":
        cos_tr = cos_tr.copy()
        if 0 <= int(idx) < cos_tr.size:
            cos_tr[int(idx)] = np.nan
    else:
        raise ValueError("side must be 'BL' or 'TR'")

    bl_valid = cos_bl[np.isfinite(cos_bl)]
    tr_valid = cos_tr[np.isfinite(cos_tr)]

    if bl_valid.size == 0 or tr_valid.size == 0:
        return dict(
            kmax=0,
            k_peak_used=[],
            k_bulk_used=[],
            BL_peak=np.nan,
            TR_peak=np.nan,
            BL_bulk=np.nan,
            TR_bulk=np.nan,
            D_peak=np.nan,
            D_bulk=np.nan,
            side=side,
            idx=int(idx),
            cos_to_BL_sorted=np.asarray([], dtype=np.float32),
            cos_to_TR_sorted=np.asarray([], dtype=np.float32),
            BL_curve=np.asarray([], dtype=np.float32),
            TR_curve=np.asarray([], dtype=np.float32),
            diff_curve=np.asarray([], dtype=np.float32),
        )

    bl_sorted = np.sort(bl_valid)[::-1]
    tr_sorted = np.sort(tr_valid)[::-1]
    bl_curve = _topk_mean_curve(bl_sorted)
    tr_curve = _topk_mean_curve(tr_sorted)

    metrics = _support_metrics_from_curves(bl_curve, tr_curve, k_peak, k_bulk)
    metrics.update(
        side=side,
        idx=int(idx),
        cos_to_BL_sorted=bl_sorted,
        cos_to_TR_sorted=tr_sorted,
        BL_curve=bl_curve,
        TR_curve=tr_curve,
        diff_curve=bl_curve[:metrics["kmax"]] - tr_curve[:metrics["kmax"]],
    )
    return metrics


def compute_bl_tr_support_decisions_from_groups(
    sn_bl,
    sn_tr,
    *,
    cos_mask_adc=30.0,
    k_peak=(5, 10, 20),
    k_bulk=(50, 100, 200),
    min_bl_bulk=0.70,
    diag_eps=0.05,
):
    sn_bl = np.asarray(sn_bl, dtype=np.float32)
    sn_tr = np.asarray(sn_tr, dtype=np.float32)

    if sn_bl.ndim != 3 or sn_tr.ndim != 3:
        raise ValueError("sn_bl and sn_tr must be [C, L, N]")
    if sn_bl.shape[0] != sn_tr.shape[0] or sn_bl.shape[1] != sn_tr.shape[1]:
        raise ValueError("sn_bl and sn_tr must match on [C, L]")
    if sn_bl.shape[2] < 2 or sn_tr.shape[2] < 2:
        raise ValueError("Need at least 2 BL and 2 TR spikes for support decisions.")

    med_bl = np.median(sn_bl, axis=2).astype(np.float32)
    med_tr = np.median(sn_tr, axis=2).astype(np.float32)

    mask = (np.abs(med_bl) >= float(cos_mask_adc)) | (np.abs(med_tr) >= float(cos_mask_adc))
    if int(mask.sum()) == 0:
        raise ValueError(f"Support mask is empty for cos_mask_adc={cos_mask_adc}.")

    X_bl = _flatten_masked_snips(sn_bl, mask)
    X_tr = _flatten_masked_snips(sn_tr, mask)
    X_bl_n = _row_normalize(X_bl)
    X_tr_n = _row_normalize(X_tr)

    bl_metrics = []
    bl_labels = []
    for idx in range(X_bl_n.shape[0]):
        m = _compute_one_spike_metrics(
            v=X_bl_n[idx],
            X_bl_n=X_bl_n,
            X_tr_n=X_tr_n,
            side="BL",
            idx=idx,
            k_peak=k_peak,
            k_bulk=k_bulk,
        )
        label = _assign_support_label(m, min_bl_bulk=min_bl_bulk, diag_eps=diag_eps)
        m["label"] = label
        bl_metrics.append(m)
        bl_labels.append(label)

    tr_metrics = []
    tr_labels = []
    for idx in range(X_tr_n.shape[0]):
        m = _compute_one_spike_metrics(
            v=X_tr_n[idx],
            X_bl_n=X_bl_n,
            X_tr_n=X_tr_n,
            side="TR",
            idx=idx,
            k_peak=k_peak,
            k_bulk=k_bulk,
        )
        label = _assign_support_label(m, min_bl_bulk=min_bl_bulk, diag_eps=diag_eps)
        m["label"] = label
        tr_metrics.append(m)
        tr_labels.append(label)

    def _count_labels(lbls):
        lbls = np.asarray(lbls, dtype=object)
        return dict(
            LH=int(np.sum(lbls == "LH")),
            soup=int(np.sum(lbls == "soup")),
            uncertain_boundary=int(np.sum(lbls == "uncertain_boundary")),
            uncertain_lowBL=int(np.sum(lbls == "uncertain_lowBL")),
            total=int(lbls.size),
        )

    return dict(
        params=dict(
            COS_MASK_ADC=float(cos_mask_adc),
            K_PEAK=list(k_peak),
            K_BULK=list(k_bulk),
            MIN_BL_BULK=float(min_bl_bulk),
            DIAG_EPS=float(diag_eps),
        ),
        med_bl=med_bl,
        med_tr=med_tr,
        mask=mask,
        bl_metrics=bl_metrics,
        tr_metrics=tr_metrics,
        bl_labels=np.asarray(bl_labels, dtype=object),
        tr_labels=np.asarray(tr_labels, dtype=object),
        bl_counts=_count_labels(bl_labels),
        tr_counts=_count_labels(tr_labels),
    )


def plot_bl_tr_support_scatter(decision_out, title=None):
    bl_metrics = decision_out["bl_metrics"]
    tr_metrics = decision_out["tr_metrics"]

    bl_x = np.array([m["BL_bulk"] for m in bl_metrics], dtype=float)
    bl_y = np.array([m["TR_bulk"] for m in bl_metrics], dtype=float)
    tr_x = np.array([m["BL_bulk"] for m in tr_metrics], dtype=float)
    tr_y = np.array([m["TR_bulk"] for m in tr_metrics], dtype=float)

    p = decision_out["params"]
    min_bl_bulk = float(p["MIN_BL_BULK"])
    diag_eps = float(p["DIAG_EPS"])

    plt.figure(figsize=(7, 7))
    plt.scatter(bl_x, bl_y, s=20, alpha=0.70, label="BL spikes")
    plt.scatter(tr_x, tr_y, s=20, alpha=0.70, label="TR spikes")

    xx0 = min(np.min(bl_x), np.min(tr_x), np.min(bl_y), np.min(tr_y)) - 0.02
    xx1 = max(np.max(bl_x), np.max(tr_x), np.max(bl_y), np.max(tr_y)) + 0.02
    xx = np.linspace(xx0, xx1, 200)
    plt.plot(xx, xx, "--", linewidth=1, alpha=0.8, label="BL_bulk = TR_bulk")
    plt.axvline(min_bl_bulk, linestyle=":", linewidth=1, alpha=0.8)
    plt.axhline(min_bl_bulk, linestyle=":", linewidth=1, alpha=0.8)
    plt.fill_between(xx, xx - diag_eps, xx + diag_eps, alpha=0.08)

    plt.xlabel("BL bulk support")
    plt.ylabel("TR bulk support")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.title("BL vs TR bulk support" if title is None else title)
    plt.tight_layout()
    plt.savefig(f'{title if title is not None else "bl_tr_support_scatter"}.png', bbox_inches='tight')
    plt.close()


def _norm_cdf(x, mu, sig):
    z = (x - mu) / (sig * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def choose_adaptive_km_window(
    raw_data,
    left_times,
    *,
    probe_n=500,
    probe_win=(-50, 100),
    fallback_win=(-20, 40),
    time_amp_thr=30.0,
    ch_ptp_thr=30.0,
    pad_left=3,
    pad_right=3,
    min_pre=16,
    min_post=28,
    rng=None,
):
    """
    Use a small long-window probe set from LEFT spikes to choose an adaptive
    downstream snippet window. Step1 stays fixed; everything after this can use
    the returned km_win.
    """
    left_times = np.asarray(left_times, dtype=np.int64)
    fallback_win = (int(fallback_win[0]), int(fallback_win[1]))
    probe_win = (int(probe_win[0]), int(probe_win[1]))

    if left_times.size == 0:
        return fallback_win, dict(
            status="no_left_times",
            probe_n_req=0,
            probe_n_valid=0,
            n_ch_keep=0,
            left_rel=None,
            right_rel=None,
        )

    if rng is None:
        rng = np.random.RandomState(0)

    if left_times.size > int(probe_n):
        pick = left_times[rng.choice(left_times.size, int(probe_n), replace=False)]
    else:
        pick = left_times

    snips_probe, valid_probe = extract_snippets_fast_ram(
        raw_data,
        pick,
        window=probe_win,
        selected_channels=np.arange(raw_data.shape[1], dtype=np.int32),
    )

    if snips_probe.shape[2] == 0:
        return fallback_win, dict(
            status="probe_empty",
            probe_n_req=int(pick.size),
            probe_n_valid=0,
            n_ch_keep=0,
            left_rel=None,
            right_rel=None,
        )

    ei_probe = snips_probe.mean(axis=2).astype(np.float32)
    p2p = ei_probe.max(axis=1) - ei_probe.min(axis=1)

    ch_keep = np.flatnonzero(p2p >= float(ch_ptp_thr))
    if ch_keep.size == 0:
        ch_keep = np.argsort(p2p)[-16:]
        ch_keep.sort()

    time_keep = np.any(np.abs(ei_probe[ch_keep, :]) >= float(time_amp_thr), axis=0)
    if not np.any(time_keep):
        return fallback_win, dict(
            status="no_time_support",
            probe_n_req=int(pick.size),
            probe_n_valid=int(snips_probe.shape[2]),
            n_ch_keep=int(ch_keep.size),
            left_rel=None,
            right_rel=None,
        )

    idx = np.flatnonzero(time_keep)
    left_rel = int(probe_win[0] + idx[0])
    right_rel = int(probe_win[0] + idx[-1])

    km_pre = int(max(probe_win[0], min(-int(min_pre), left_rel - int(pad_left))))
    km_post = int(min(probe_win[1], max(int(min_post), right_rel + int(pad_right))))

    if km_pre >= km_post:
        return fallback_win, dict(
            status="bad_window",
            probe_n_req=int(pick.size),
            probe_n_valid=int(snips_probe.shape[2]),
            n_ch_keep=int(ch_keep.size),
            left_rel=int(left_rel),
            right_rel=int(right_rel),
        )

    return (km_pre, km_post), dict(
        status="ok",
        probe_n_req=int(pick.size),
        probe_n_valid=int(snips_probe.shape[2]),
        n_ch_keep=int(ch_keep.size),
        left_rel=int(left_rel),
        right_rel=int(right_rel),
    )

def compute_left_isi_pairs_10_30(step1):
    lt = np.asarray(step1.get("left_times", []), dtype=np.int64)
    lt = np.sort(lt)
    if lt.size < 2:
        return 0
    d = np.diff(lt)
    return int(np.sum((d >= 10) & (d <= 30)))


def plot_amp_hist(step1, ch, ycap=5000):
    edges = np.asarray(step1["amp_hist_edges"], dtype=np.float64)
    counts = np.asarray(step1["amp_hist_counts"], dtype=np.float64)

    plt.figure(figsize=(12, 3))
    plt.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
    plt.axvspan(step1["valley_low"], step1["valley_high"], color="r", alpha=0.25)
    plt.ylim(0, ycap)
    plt.title(
        f"CH {ch}: amplitude histogram | left={step1['left_count']} valley={step1['valley_count']}"
    )
    plt.xlabel("minima amplitude (ADC)")
    plt.ylabel("count")
    plt.grid(alpha=0.20)
    plt.tight_layout()
    plt.savefig(f"ch{ch}_amp_hist.png", bbox_inches="tight")
    plt.close()


def plot_final_ei(ei, ref_channel, title, ei_positions):
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    pew.plot_ei_waveforms(
        ei,
        positions=ei_positions,
        ref_channel=int(ref_channel),
        scale=70.0,
        ax=ax,
        colors="black",
        alpha=1.0,
        linewidth=0.8,
        box_height=1.0,
        box_width=50.0,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_final_ei.png", bbox_inches="tight")
    plt.close()


def shift_ei(ei, lag):
    ei = np.asarray(ei)
    out = np.zeros_like(ei)
    if lag == 0:
        out[:] = ei
    elif lag > 0:
        out[:, lag:] = ei[:, :-lag]
    else:
        s = -lag
        out[:, :ei.shape[1] - s] = ei[:, s:]
    return out


def support_from_ei(ei, support_rel=0.10, support_abs=30.0):
    p2p = np.ptp(ei, axis=1)
    thr = max(float(support_abs), float(support_rel) * float(p2p.max()))
    S = p2p >= thr
    return S, p2p, thr


def best_lag_on_support(X, Y, S, max_lag=3, time_keep_rel=0.10):
    Xs = X[S, :]
    env = np.max(np.abs(Xs), axis=0) if Xs.size else np.max(np.abs(X), axis=0)
    tthr = float(time_keep_rel) * float(env.max() + 1e-12)
    Tmask = env >= tthr
    if not np.any(Tmask):
        Tmask = np.ones(X.shape[1], dtype=bool)

    best = dict(lag=0, cos=-np.inf, T=Tmask)
    A = X[S, :][:, Tmask].ravel()
    nA = np.linalg.norm(A) + 1e-12
    for lag in range(-int(max_lag), int(max_lag) + 1):
        Ys = shift_ei(Y, lag)
        B = Ys[S, :][:, Tmask].ravel()
        nB = np.linalg.norm(B) + 1e-12
        cos = float((A @ B) / (nA * nB))
        if cos > best["cos"]:
            best = dict(lag=int(lag), cos=cos, T=Tmask)
    return best


def containment_metrics(X, Y, *, max_lag=3, support_rel=0.10, support_abs=30.0, time_keep_rel=0.10):
    S, p2pX, thr = support_from_ei(X, support_rel=support_rel, support_abs=support_abs)
    best = best_lag_on_support(X, Y, S, max_lag=max_lag, time_keep_rel=time_keep_rel)
    lag = best["lag"]
    Yal = shift_ei(Y, lag)
    Tmask = best["T"]

    A = X[S, :][:, Tmask].ravel()
    B = Yal[S, :][:, Tmask].ravel()
    alpha = float((A @ B) / ((A @ A) + 1e-12))

    R = Yal - alpha * X
    Yin  = Yal[S, :]
    Rin  = R[S, :]
    Rout = R[~S, :]

    Ein = float(np.linalg.norm(Rin))
    Eout = float(np.linalg.norm(Rout))

    frac_in  = float(np.linalg.norm(Rin) / (np.linalg.norm(Yin) + 1e-12))
    frac_all = float(np.linalg.norm(R) / (np.linalg.norm(Yal) + 1e-12))
    out_in   = float(Eout / (Ein + 1e-12))

    return dict(
        lag=lag,
        cos_on_support=float(best["cos"]),
        alpha=alpha,
        support_n=int(np.sum(S)),
        support_thr=float(thr),
        frac_in=frac_in,
        frac_all=frac_all,
        out_in=out_in,
    )


def verdict_from_kmeans(
    ei0,
    ei1,
    *,
    max_lag=3,
    support_rel=0.10,
    support_abs=30.0,
    time_keep_rel=0.10,
    frac_in_thr=0.20,
    out_in_ratio_thr=2.0,
    resid_frac_min=0.08,
    shared_cos_thr=0.95,
    shared_alpha_thr=0.95,
):
    m01 = containment_metrics(
        ei0, ei1,
        max_lag=max_lag,
        support_rel=support_rel,
        support_abs=support_abs,
        time_keep_rel=time_keep_rel,
    )
    m10 = containment_metrics(
        ei1, ei0,
        max_lag=max_lag,
        support_rel=support_rel,
        support_abs=support_abs,
        time_keep_rel=time_keep_rel,
    )

    shared01 = (m01["cos_on_support"] >= shared_cos_thr) and (m01["alpha"] >= shared_alpha_thr)
    shared10 = (m10["cos_on_support"] >= shared_cos_thr) and (m10["alpha"] >= shared_alpha_thr)
    shared_core = bool(shared01 or shared10)

    def is_contained(m):
        return (m["frac_in"] <= frac_in_thr) and (m["out_in"] >= out_in_ratio_thr)

    c01 = is_contained(m01)
    c10 = is_contained(m10)

    if shared_core and c01 and c10 and (m01["frac_all"] < resid_frac_min) and (m10["frac_all"] < resid_frac_min):
        verdict = "SAME UNIT split (amplitude/drift)"
        proceed = True
    elif shared_core and ((c01 and not c10) or (c10 and not c01)):
        verdict = "AB-SHARD-like (shared core)"
        proceed = True
    elif shared_core:
        verdict = "SHARED-CORE (overlap/AA/complex)"
        proceed = True
    elif (not c01) and (not c10):
        verdict = "TWO-UNITS-like (reject)"
        proceed = False
    else:
        verdict = "AMBIGUOUS (reject)"
        proceed = False

    return dict(
        verdict=verdict,
        proceed=bool(proceed),
        shared_core=shared_core,
        shared_dirs=dict(ei0_to_ei1=bool(shared01), ei1_to_ei0=bool(shared10)),
        m01=m01,
        m10=m10,
    )


def kmeans_pair_metrics(
    ei0,
    ei1,
    *,
    support_rel=0.10,
    support_abs=30.0,
    max_lag=1,
):
    S0, p2p0, thr0 = support_from_ei(ei0, support_rel=support_rel, support_abs=support_abs)
    S1, p2p1, thr1 = support_from_ei(ei1, support_rel=support_rel, support_abs=support_abs)

    U = S0 | S1
    if not np.any(U):
        U = (p2p0 > 0) | (p2p1 > 0)
    if not np.any(U):
        U = np.zeros(ei0.shape[0], dtype=bool)
        U[0] = True

    A = np.asarray(ei0[U, :], dtype=np.float32).ravel()
    nA = float(np.linalg.norm(A) + 1e-12)

    best = dict(lag=0, cos=-np.inf)
    for lag in range(-int(max_lag), int(max_lag) + 1):
        Y = shift_ei(ei1, lag)
        B = np.asarray(Y[U, :], dtype=np.float32).ravel()
        nB = float(np.linalg.norm(B) + 1e-12)
        cos = float((A @ B) / (nA * nB))
        if cos > best["cos"]:
            best = dict(lag=int(lag), cos=float(cos))

    return dict(
        cos=float(best["cos"]),
        lag=int(best["lag"]),
        union_n=int(np.sum(U)),
        unique0=int(np.sum(S0 & ~S1)),
        unique1=int(np.sum(S1 & ~S0)),
        thr0=float(thr0),
        thr1=float(thr1),
    )


def kmeans_precheck_decision(
    vr,
    n0,
    n1,
    ei0,
    ei1,
    *,
    pc_var_thr=0.10,
    minor_frac_thr=0.10,
    cos_oneunit_thr=0.95,
    asym_unique_ch_min=3,
    support_rel=0.10,
    support_abs=30.0,
    cos_lag=1,
):
    vr = np.asarray(vr, dtype=np.float32).ravel()
    vr12 = vr[:2].copy()

    # 1) PCA variance rule: reject as TWO UNITS
    if np.any(vr12 > float(pc_var_thr)):
        return dict(
            decided=True,
            proceed=False,
            verdict="TWO-UNITS-like (PC variance)",
            reason="pc_var",
            detail=(
                f"expl_var=[" + ", ".join(f"{100.0 * float(v):.2f}%" for v in vr12[:2]) + "] "
                f"> {100.0 * float(pc_var_thr):.1f}%"
            ),
            pair=None,
            minor_frac=None,
        )

    # 2) Tiny secondary cluster: accept as ONE UNIT
    n_big = int(max(n0, n1))
    n_small = int(min(n0, n1))
    minor_frac = float(n_small / max(n_big, 1))

    if minor_frac < float(minor_frac_thr):
        return dict(
            decided=True,
            proceed=True,
            verdict="ONE UNIT (tiny secondary cluster)",
            reason="cluster_size",
            detail=(
                f"n0={int(n0)} n1={int(n1)} "
                f"minor_frac={100.0 * float(minor_frac):.1f}% < {100.0 * float(minor_frac_thr):.1f}%"
            ),
            pair=None,
            minor_frac=minor_frac,
        )

    # 3) High EI cosine on union of significant channels: accept as ONE UNIT
    pair = kmeans_pair_metrics(
        ei0,
        ei1,
        support_rel=support_rel,
        support_abs=support_abs,
        max_lag=cos_lag,
    )

    if pair["cos"] > float(cos_oneunit_thr):
        return dict(
            decided=True,
            proceed=True,
            verdict="ONE UNIT (high EI cosine)",
            reason="ei_cos",
            detail=f"cos={float(pair['cos']):.2f} lag={pair['lag']} union_n={pair['union_n']} > {cos_oneunit_thr:.2f}",
            pair=pair,
            minor_frac=minor_frac,
        )

    # 4) Symmetric unique significant channels: reject as TWO UNITS
    if (pair["unique0"] >= int(asym_unique_ch_min)) and (pair["unique1"] >= int(asym_unique_ch_min)):
        return dict(
            decided=True,
            proceed=False,
            verdict="TWO-UNITS-like (asymmetric significant channels)",
            reason="asym_sig_channels",
            detail=f"unique0={pair['unique0']} unique1={pair['unique1']} >= {int(asym_unique_ch_min)}",
            pair=pair,
            minor_frac=minor_frac,
        )

    # Inconclusive -> caller should fall back to verdict_from_kmeans
    vr_str = "[" + ", ".join(f"{100.0 * float(v):.2f}%" for v in vr12[:2]) + "]"

    return dict(
        decided=False,
        proceed=True,
        verdict="INCONCLUSIVE",
        reason="inconclusive",
        detail=(
            f"expl_var={vr_str} "
            f"minor_frac={100.0 * float(minor_frac):.1f}% "
            f"cos={float(pair['cos']):.2f} lag={pair['lag']} "
            f"unique0={pair['unique0']} unique1={pair['unique1']}"
        ),
        pair=pair,
        minor_frac=minor_frac,
    )




def stable_unique_int64(x):
    x = np.asarray(x, dtype=np.int64).ravel()
    if x.size == 0:
        return x
    x_sorted = np.sort(x)
    keep = np.ones(x_sorted.size, dtype=bool)
    keep[1:] = x_sorted[1:] != x_sorted[:-1]
    return x_sorted[keep]


def build_template_bank(step4, *, raw_mod, win, sr, n_bins=5, spikes_per=5000, ei_p2p_thr=50.0):
    main_ch = int(step4.get("main_ch", 0))
    final_ei = np.asarray(step4["final_ei"], dtype=np.float32)
    ei_p2p = np.ptp(final_ei, axis=1)

    chan_sel = np.flatnonzero(ei_p2p >= float(ei_p2p_thr)).astype(np.int32)
    if chan_sel.size == 0:
        raise RuntimeError(f"No channels pass EI p2p >= {ei_p2p_thr} (max={ei_p2p.max():.1f})")

    chan_sel = chan_sel[np.argsort(ei_p2p[chan_sel])[::-1]]
    if not np.any(chan_sel == main_ch):
        chan_sel = np.concatenate([np.asarray([main_ch], dtype=np.int32), chan_sel])
        chan_sel = np.unique(chan_sel)

    times_all = np.sort(np.asarray(step4.get("final_times", step4.get("final_times_used", [])), dtype=np.int64))
    if times_all.size == 0:
        raise RuntimeError("No spike times in step4.")

    sn_main, valid_times = extract_snippets_fast_ram(
        raw_mod,
        times_all,
        window=win,
        selected_channels=np.asarray([main_ch], dtype=np.int32),
    )
    valid_times = np.asarray(valid_times, dtype=np.int64)
    amps = np.ptp(sn_main[0, :, :], axis=0).astype(np.float32)

    n_bins = int(n_bins)
    spikes_per = int(spikes_per)
    if amps.size < n_bins * 10:
        raise RuntimeError(f"Too few valid snippets for template binning: Nvalid={amps.size}")

    order_desc = np.argsort(amps)[::-1]
    N = int(order_desc.size)
    bin_times = []
    bin_amp_ranges = []

    if N >= n_bins * spikes_per:
        starts = np.round(np.linspace(0, N - spikes_per, n_bins)).astype(int) if n_bins > 1 else np.array([0], dtype=int)
        for s in starts:
            idx = order_desc[s:s + spikes_per]
            tbin = valid_times[idx]
            abin = amps[idx]
            bin_times.append(tbin)
            bin_amp_ranges.append((float(np.min(abin)), float(np.max(abin))))
    else:
        groups = np.array_split(order_desc, n_bins)
        for g in groups:
            tbin = valid_times[g]
            abin = amps[g]
            bin_times.append(tbin)
            if abin.size:
                bin_amp_ranges.append((float(np.min(abin)), float(np.max(abin))))
            else:
                bin_amp_ranges.append((np.nan, np.nan))

    if any(t.size == 0 for t in bin_times):
        raise RuntimeError("Empty template bin; reduce n_bins or get more spikes.")

    templates = []
    for tbin in bin_times:
        sn, _ = extract_snippets_fast_ram(raw_mod, tbin, window=win, selected_channels=chan_sel)
        templates.append(np.median(sn, axis=2).astype(np.float32))
    templates = np.stack(templates, axis=0)

    return dict(
        main_ch=int(main_ch),
        win=tuple(win),
        sample_rate_hz=float(sr),
        channels=chan_sel,
        ei_p2p=ei_p2p[chan_sel],
        bin_amp_ranges=bin_amp_ranges,
        templates=templates,
    )


def build_median_ei_from_times(raw_mod, times, window, n_channels):
    times = np.asarray(times, dtype=np.int64)
    times = np.sort(np.unique(times))
    if times.size == 0:
        return None, 0

    sn, _ = extract_snippets_fast_ram(
        raw_mod,
        times,
        window=window,
        selected_channels=np.arange(int(n_channels), dtype=np.int32),
    )
    if sn.shape[2] == 0:
        return None, 0
    return np.median(sn, axis=2).astype(np.float32), int(sn.shape[2])



def subtract_lh_masked(
    raw_mod,
    spike_times,
    bank,
    *,
    time_keep_frac=0.08,
    weight_power=1.0,
    use_ch_weight_sub=True,
    min_sep_ms=1.0,
    cos_thr_sub=0.90,
    jmax_sub=3,
    batch_sub=256,
):
    templates = np.asarray(bank["templates"], dtype=np.float32)
    chs = np.asarray(bank["channels"], dtype=np.int32)
    win_ = tuple(bank["win"])
    sr = float(bank["sample_rate_hz"])
    B, Kk, L = templates.shape
    templates_i16 = np.rint(templates).astype(np.int16)

    ref = np.median(templates, axis=0)
    absref = np.abs(ref)

    time_env = absref.max(axis=0)
    t_thr = float(time_keep_frac) * time_env.max()
    keep_t = time_env >= t_thr

    if bool(use_ch_weight_sub):
        ch_env = np.ptp(ref, axis=1)
        ch_w = ch_env / (ch_env.max() + 1e-12)
    else:
        ch_w = np.ones(Kk, dtype=np.float32)

    W = (absref ** float(weight_power)) * ch_w[:, None]
    W[:, ~keep_t] = 0.0

    mask_idx = np.flatnonzero(W.ravel() > 0)
    w_mask = W.ravel()[mask_idx].astype(np.float32)

    tmpl_flat_full = templates.reshape(B, -1).astype(np.float32)
    tmplW = tmpl_flat_full[:, mask_idx] * w_mask[None, :]
    tmplW_norm = np.linalg.norm(tmplW, axis=1) + 1e-12

    main_ch = int(bank.get("main_ch", 0))
    if not np.any(chs == main_ch):
        raise RuntimeError(f"main_ch={main_ch} not found in bank channels.")

    main_ci = int(np.where(chs == main_ch)[0][0])
    center_idx = int(-win_[0])
    tmin_main = int(np.argmin(ref[main_ci, :]))
    dt_main = int(tmin_main - center_idx)

    win_ext = (win_[0] - int(jmax_sub), win_[1] + int(jmax_sub))
    spike_times = np.asarray(spike_times, dtype=np.int64)
    spike_times = np.sort(np.unique(spike_times))
    if spike_times.size == 0:
        return None

    used_t_all = []
    best_b_all = []
    best_j_all = []
    best_c_all = []

    tmp_sn, _ = extract_snippets_fast_ram(raw_mod, spike_times[:1], window=win_ext, selected_channels=chs)
    if tmp_sn.shape[2] == 0:
        return None
    Lext0 = tmp_sn.shape[1]
    anchor0 = (Lext0 - L) // 2

    for s0 in range(0, spike_times.size, int(batch_sub)):
        t_batch = spike_times[s0:s0 + int(batch_sub)]
        sn_ext, t_valid = extract_snippets_fast_ram(raw_mod, t_batch, window=win_ext, selected_channels=chs)
        t_valid = np.asarray(t_valid, dtype=np.int64)
        N = sn_ext.shape[2]
        if N == 0:
            continue

        Lext = sn_ext.shape[1]
        anchor = (Lext - L) // 2

        best_cos = np.full(N, -np.inf, dtype=np.float32)
        best_b = np.zeros(N, dtype=np.int16)
        best_j = np.zeros(N, dtype=np.int8)

        for j in range(-int(jmax_sub), int(jmax_sub) + 1):
            start = anchor + j
            seg = sn_ext[:, start:start + L, :].astype(np.float32)
            if seg.shape[1] != L:
                continue

            Xfull = seg.transpose(2, 0, 1).reshape(N, -1)
            Xw = Xfull[:, mask_idx] * w_mask[None, :]
            Xn = np.linalg.norm(Xw, axis=1) + 1e-12
            dots = Xw @ tmplW.T
            cos = dots / (Xn[:, None] * tmplW_norm[None, :])

            b = np.argmax(cos, axis=1).astype(np.int16)
            c = cos[np.arange(N), b].astype(np.float32)

            upd = c > best_cos
            best_cos[upd] = c[upd]
            best_b[upd] = b[upd]
            best_j[upd] = np.int8(j)

        used_t_all.append(t_valid)
        best_b_all.append(best_b)
        best_j_all.append(best_j)
        best_c_all.append(best_cos)

    if len(used_t_all) == 0:
        return None

    used_t_all = np.concatenate(used_t_all).astype(np.int64)
    best_b_all = np.concatenate(best_b_all).astype(np.int16)
    best_j_all = np.concatenate(best_j_all).astype(np.int8)
    best_c_all = np.concatenate(best_c_all).astype(np.float32)

    t_center_all = used_t_all + best_j_all.astype(np.int64)
    did_sub = best_c_all >= float(cos_thr_sub)

    min_sep_samp = int(round((float(min_sep_ms) / 1000.0) * sr))
    accept_idx = np.flatnonzero(did_sub)
    if accept_idx.size > 1 and min_sep_samp > 0:
        accept_sorted = accept_idx[np.argsort(t_center_all[accept_idx])]
        tacc = t_center_all[accept_sorted]
        i = 0
        while i < accept_sorted.size:
            j = i
            while (j + 1 < accept_sorted.size) and ((tacc[j + 1] - tacc[j]) < min_sep_samp):
                j += 1
            if j > i:
                cluster = accept_sorted[i:j + 1]
                keep = cluster[np.argmax(best_c_all[cluster])]
                drop = cluster[cluster != keep]
                did_sub[drop] = False
            i = j + 1

    keep_idx = np.flatnonzero(did_sub)
    keep_idx = keep_idx[np.argsort(t_center_all[keep_idx])]

    for idx in keep_idx:
        t = int(used_t_all[idx])
        b = int(best_b_all[idx])
        j = int(best_j_all[idx])
        t0 = t + win_ext[0] + (anchor0 + j)
        t1 = t0 + L
        tmpl = templates_i16[b]
        for ci, ch_local in enumerate(chs):
            raw_mod[t0:t1, int(ch_local)] -= tmpl[ci, :]

    spike_times_center = np.sort(t_center_all[did_sub])
    spike_times_main = np.sort((t_center_all + dt_main)[did_sub])

    return dict(
        main_ch=int(main_ch),
        dt_main=int(dt_main),
        spike_times_center=spike_times_center,
        spike_times_main=spike_times_main,
        n_sub=int(did_sub.sum()),
        best_cosine=best_c_all,
        best_jitter=best_j_all,
        did_subtract=did_sub,
    )


def build_bl_tr_probe_times(
    raw_mod,
    left_times,
    tr_candidate_times,
    *,
    main_ch,
    win,
    n_per_side=2000,
):
    left_times = np.sort(np.unique(np.asarray(left_times, dtype=np.int64)))
    tr_candidate_times = np.sort(np.unique(np.asarray(tr_candidate_times, dtype=np.int64)))

    if left_times.size == 0:
        return dict(ok=False, reason="no_left_times")

    sn_left, left_valid = extract_snippets_fast_ram(
        raw_mod,
        left_times,
        window=win,
        selected_channels=np.asarray([int(main_ch)], dtype=np.int32),
    )
    left_valid = np.asarray(left_valid, dtype=np.int64)
    if sn_left.shape[2] == 0:
        return dict(ok=False, reason="no_valid_left_snips")

    amp_left = np.max(np.abs(sn_left[0, :, :]), axis=0).astype(np.float32)
    order_left = np.argsort(amp_left)  # weakest first
    n_left_pick = int(min(int(n_per_side), order_left.size))
    bl_times = left_valid[order_left[:n_left_pick]].astype(np.int64)
    tl_times = left_valid[order_left[-n_left_pick:]].astype(np.int64)

    if tr_candidate_times.size == 0:
        return dict(
            ok=True,
            bl_times=bl_times,
            tl_times=tl_times,
            tr_times=np.asarray([], dtype=np.int64),
        )

    sn_right, right_valid = extract_snippets_fast_ram(
        raw_mod,
        tr_candidate_times,
        window=win,
        selected_channels=np.asarray([int(main_ch)], dtype=np.int32),
    )
    right_valid = np.asarray(right_valid, dtype=np.int64)
    if sn_right.shape[2] == 0:
        tr_times = np.asarray([], dtype=np.int64)
    else:
        amp_right = np.max(np.abs(sn_right[0, :, :]), axis=0).astype(np.float32)
        order_right = np.argsort(amp_right)[::-1]  # strongest first
        n_right_pick = int(min(int(n_per_side), order_right.size))
        tr_times = right_valid[order_right[:n_right_pick]].astype(np.int64)

    return dict(
        ok=True,
        bl_times=bl_times,
        tl_times=tl_times,
        tr_times=tr_times,
    )


def run_bl_tr_support_for_channel(
    raw_mod,
    ei_info,
    left_times,
    tr_candidate_times,
    *,
    ch,
    win,
    n_probe_per_side=2000,
    n_top_channels=12,
    cos_mask_adc=30.0,
    k_peak=(5, 10, 20),
    k_bulk=(50, 100, 200),
    min_bl_bulk=0.70,
    diag_eps=0.05,
):
    main_ch = int(ch)
    probe = build_bl_tr_probe_times(
        raw_mod,
        left_times,
        tr_candidate_times,
        main_ch=main_ch,
        win=win,
        n_per_side=n_probe_per_side,
    )
    if not probe.get("ok", False):
        return dict(ok=False, reason=probe.get("reason", "probe_build_failed"))

    bl_times = np.asarray(probe["bl_times"], dtype=np.int64)
    tl_times = np.asarray(probe["tl_times"], dtype=np.int64)
    tr_times = np.asarray(probe["tr_times"], dtype=np.int64)

    if bl_times.size < 2:
        return dict(ok=False, reason="too_few_bl_probe")
    if tr_times.size < 2:
        return dict(
            ok=False,
            reason="too_few_tr_probe",
            probe=dict(bl_times=bl_times, tl_times=tl_times, tr_times=tr_times),
        )

    final_ei = np.asarray(ei_info["final_ei"], dtype=np.float32)
    chan_p2p = np.ptp(final_ei, axis=1)
    topn = int(min(int(n_top_channels), chan_p2p.size))
    top_ch = np.argsort(chan_p2p)[-topn:][::-1].astype(np.int32)

    sn_bl, bl_valid = extract_snippets_fast_ram(raw_mod, bl_times, window=win, selected_channels=top_ch)
    sn_tr, tr_valid = extract_snippets_fast_ram(raw_mod, tr_times, window=win, selected_channels=top_ch)
    bl_valid = np.asarray(bl_valid, dtype=np.int64)
    tr_valid = np.asarray(tr_valid, dtype=np.int64)

    if sn_bl.shape[2] < 2 or sn_tr.shape[2] < 2:
        return dict(ok=False, reason="too_few_valid_snips_after_extract")

    decision_out = compute_bl_tr_support_decisions_from_groups(
        sn_bl,
        sn_tr,
        cos_mask_adc=cos_mask_adc,
        k_peak=k_peak,
        k_bulk=k_bulk,
        min_bl_bulk=min_bl_bulk,
        diag_eps=diag_eps,
    )

    bl_labels = np.asarray(decision_out["bl_labels"], dtype=object)
    tr_labels = np.asarray(decision_out["tr_labels"], dtype=object)

    bl_keep_mask = (bl_labels == "LH")
    bl_uncertain_mask = np.isin(bl_labels, np.asarray(["uncertain_boundary", "uncertain_lowBL"], dtype=object))
    tr_keep_mask = (tr_labels == "LH")
    tr_uncertain_mask = np.isin(tr_labels, np.asarray(["uncertain_boundary", "uncertain_lowBL"], dtype=object))

    bl_reject_mask = ~(bl_keep_mask | bl_uncertain_mask)
    tr_reject_mask = ~(tr_keep_mask | tr_uncertain_mask)

    return dict(
        ok=True,
        probe=dict(
            bl_times=bl_valid,
            tl_times=tl_times,
            tr_times=tr_valid,
            top_ch=top_ch,
            main_ch=int(main_ch),
        ),
        decision_out=decision_out,
        bl_keep_times=np.asarray(bl_valid[bl_keep_mask], dtype=np.int64),
        bl_uncertain_times=np.asarray(bl_valid[bl_uncertain_mask], dtype=np.int64),
        tr_keep_times=np.asarray(tr_valid[tr_keep_mask], dtype=np.int64),
        tr_uncertain_times=np.asarray(tr_valid[tr_uncertain_mask], dtype=np.int64),
        bl_reject_times=np.asarray(bl_valid[bl_reject_mask], dtype=np.int64),
        tr_reject_times=np.asarray(tr_valid[tr_reject_mask], dtype=np.int64),
    )


def plot_kmeans_pc_scatter(km_plot, title=None):
    if km_plot is None:
        return
    Xpc = np.asarray(km_plot["Xpc"], dtype=np.float32)
    lab = np.asarray(km_plot["labels"], dtype=np.int32)
    vr = np.asarray(km_plot["vr"], dtype=np.float32)

    plt.figure(figsize=(4.8, 4.8))
    for k in np.unique(lab):
        m = (lab == k)
        plt.scatter(Xpc[m, 0], Xpc[m, 1], s=12, alpha=0.75, label=f"cluster {int(k)}")
    plt.xlabel(f"PC1 ({vr[0] * 100.0:.1f}% var)" if vr.size >= 1 else "PC1")
    plt.ylabel(f"PC2 ({vr[1] * 100.0:.1f}% var)" if vr.size >= 2 else "PC2")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.title("k=2 PCA scatter" if title is None else title)
    plt.tight_layout()
    plt.savefig(f'{title if title is not None else "kmeans_pc_scatter"}.png', bbox_inches='tight')
    plt.close()



def _new_audit_record(ch):
    return dict(
        ch=int(ch),
        status="started",               # started / success / rejected / exception
        final_reason=None,              # e.g. valley_not_accepted, valley_count>max, SUCCESS
        error_type=None,
        error_msg=None,

        # step1 summary
        step1_accepted=None,
        left_count=None,
        valley_count=None,
        required_ratio=None,
        valley_low=None,
        valley_high=None,

        # adaptive window summary
        km_win_pre=None,
        km_win_post=None,
        km_probe_status=None,
        km_probe_left_rel=None,
        km_probe_right_rel=None,
        km_probe_n_req=None,
        km_probe_n_valid=None,
        km_probe_n_ch=None,

        # timings we explicitly care about
        t_step1_s=np.nan,
        t_km_window_s=np.nan,
        t_kmeans_extract_s=np.nan,
        t_channel_total_s=np.nan,

        # downstream summary
        isi_pairs_10_30=None,
        kmeans_verdict=None,
        kmeans_reason=None,
        main_ch=None,
        final_times_n=None,
        final_spikes_main_n=None,
        valley0_n=None,

        bl_probe_to_soup=None,
        bl_probe_to_unc=None,
        tr_probe_to_lh=None,

        used_support_filter=False,
        unit_index=None,
    )

def _set_reject(audit, reason):
    audit["status"] = "rejected"
    audit["final_reason"] = str(reason)

def _set_success(audit):
    audit["status"] = "success"
    audit["final_reason"] = "SUCCESS"

# def _append_audit_record(audit):
    # lh_channel_audit.append(audit)

def _build_valley_overflow_summary(audit_records, valley_limit, bin_step=200, hard_cap=1000):
    vals = [
        int(rec["valley_count"])
        for rec in audit_records
        if rec.get("final_reason") == "valley_count>max" and rec.get("valley_count") is not None
    ]
    out = []
    if len(vals) == 0:
        return out

    lo = int(valley_limit) + 1
    while lo <= int(hard_cap):
        hi = min(lo + int(bin_step) - 1, int(hard_cap))
        cnt = int(np.sum((np.asarray(vals) >= lo) & (np.asarray(vals) <= hi)))
        out.append((lo, hi, cnt))
        lo += int(bin_step)

    gt_cap = int(np.sum(np.asarray(vals) > int(hard_cap)))
    out.append((int(hard_cap) + 1, None, gt_cap))
    return out


def _channel_precompute_worker(ch):
    raw_mod = _PARALLEL_RAW_MOD
    cfg = _PARALLEL_CFG

    if raw_mod is None or cfg is None:
        raise RuntimeError("Parallel worker not initialized with cfg/raw_mod")

    C = int(cfg["C"])
    sr = float(cfg["sr"])
    win = tuple(cfg["win"])

    channel_stage_times = {}
    ch_t0 = time.perf_counter()
    audit = _new_audit_record(ch)

    rng = np.random.RandomState(int(cfg["RNG_SEED"]) + int(ch))

    try:
        step1 = _timed_call(
            "step1.find_valley_and_times",
            find_valley_and_times,
            raw_mod,
            ch,
            channel_tag=ch,
            timings=channel_stage_times,
            window=win,
            start=0,
            stop=None,
            bin_width=float(cfg["BIN_WIDTH"]),
            valley_bins=int(cfg["VALLEY_BINS"]),
            min_valid_count=int(cfg["MIN_VALID_COUNT"]),
            ratio_base=float(cfg["RATIO_BASE"]),
            ratio_step=float(cfg["RATIO_STEP"]),
            ratio_floor=float(cfg["RATIO_FLOOR"]),
            ratio_cap=float(cfg["RATIO_CAP"]),
            right_k=int(cfg["SUPPORT_N_PROBE_PER_SIDE"]),
            min_trough=-2500,
        )

        audit["step1_accepted"] = bool(step1.get("accepted", False))
        audit["left_count"] = int(step1.get("left_count", 0))
        audit["valley_count"] = int(step1.get("valley_count", 0))
        audit["required_ratio"] = (
            None if step1.get("required_ratio", None) is None else float(step1.get("required_ratio"))
        )
        audit["valley_low"] = step1.get("valley_low", None)
        audit["valley_high"] = step1.get("valley_high", None)
        audit["t_step1_s"] = channel_stage_times.get("step1.find_valley_and_times", np.nan)

        if int(step1.get("valley_count", 0)) > int(cfg["MAX_VALLEY_COUNT"]):
            _set_reject(audit, "valley_count>max")
            return dict(status="rejected", audit=audit)

        if not step1.get("accepted", False):
            _set_reject(audit, "valley_not_accepted")
            return dict(status="rejected", audit=audit)

        left_times = np.asarray(step1["left_times"], dtype=np.int64)

        km_win, km_win_info = _timed_call(
            "step2.choose_adaptive_km_window",
            choose_adaptive_km_window,
            raw_mod,
            left_times,
            channel_tag=ch,
            timings=channel_stage_times,
            probe_n=int(cfg["KM_PROBE_N"]),
            probe_win=tuple(cfg["KM_PROBE_WIN"]),
            fallback_win=win,
            time_amp_thr=float(cfg["KM_PROBE_TIME_AMP_THR"]),
            ch_ptp_thr=float(cfg["KM_PROBE_CH_P2P_THR"]),
            pad_left=int(cfg["KM_WIN_PAD_LEFT"]),
            pad_right=int(cfg["KM_WIN_PAD_RIGHT"]),
            min_pre=int(cfg["KM_WIN_MIN_PRE"]),
            min_post=int(cfg["KM_WIN_MIN_POST"]),
            rng=rng,
        )

        audit["km_win_pre"] = int(km_win[0])
        audit["km_win_post"] = int(km_win[1])
        audit["km_probe_status"] = str(km_win_info.get("status", "unknown"))
        audit["km_probe_left_rel"] = km_win_info.get("left_rel", None)
        audit["km_probe_right_rel"] = km_win_info.get("right_rel", None)
        audit["km_probe_n_req"] = km_win_info.get("probe_n_req", None)
        audit["km_probe_n_valid"] = km_win_info.get("probe_n_valid", None)
        audit["km_probe_n_ch"] = km_win_info.get("n_ch_keep", None)
        audit["t_km_window_s"] = channel_stage_times.get("step2.choose_adaptive_km_window", np.nan)

        isi_pairs_10_30 = _timed_call(
            "step2.compute_left_isi_pairs_10_30",
            compute_left_isi_pairs_10_30,
            step1,
            channel_tag=ch,
            timings=channel_stage_times,
        )
        audit["isi_pairs_10_30"] = int(isi_pairs_10_30)

        if isi_pairs_10_30 > int(cfg["ISI_10_30_MAX"]):
            _set_reject(audit, "abort_ISI10_30")
            return dict(status="rejected", audit=audit)

        km_info = None
        km_plot = None
        base_ei = None

        if bool(cfg["DO_KMEANS"]):
            left_times = np.asarray(step1["left_times"], dtype=np.int64)

            if left_times.size > int(cfg["SUBSAMPLE_MAX"]):
                pick = left_times[rng.choice(left_times.size, int(cfg["SUBSAMPLE_MAX"]), replace=False)]
            else:
                pick = left_times

            snips_km, _ = _timed_call(
                "step3a.extract_snippets_fast_ram[kmeans]",
                extract_snippets_fast_ram,
                raw_mod,
                pick,
                channel_tag=ch,
                timings=channel_stage_times,
                window=km_win,
                selected_channels=np.arange(C, dtype=np.int32),
            )
            audit["t_kmeans_extract_s"] = channel_stage_times.get(
                "step3a.extract_snippets_fast_ram[kmeans]", np.nan
            )

            N = snips_km.shape[2]
            if N == 0:
                _set_reject(audit, "base_ei_empty")
                return dict(status="rejected", audit=audit)

            ei_mean = snips_km.mean(axis=2).astype(np.float32)
            base_ei = ei_mean
            rms = np.sqrt(np.mean(ei_mean ** 2, axis=1))
            sel = np.flatnonzero(rms > float(cfg["RMS_THRESH"]))
            if sel.size == 0:
                sel = np.argsort(rms)[-16:]
                sel.sort()

            X = snips_km[sel, :, :].transpose(2, 0, 1).reshape(
                N, sel.size * snips_km.shape[1]
            ).astype(np.float32)

            n_pc = int(min(int(cfg["N_PC"]), X.shape[0], X.shape[1]))
            pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0)
            Xpc = _timed_call(
                "step3b.PCA.fit_transform",
                pca.fit_transform,
                X,
                channel_tag=ch,
                timings=channel_stage_times,
            )
            vr = pca.explained_variance_ratio_

            km = KMeans(n_clusters=2, n_init=50, random_state=0)
            lab = _timed_call(
                "step3c.KMeans.fit_predict",
                km.fit_predict,
                Xpc,
                channel_tag=ch,
                timings=channel_stage_times,
            )

            n0 = int(np.sum(lab == 0))
            n1 = int(np.sum(lab == 1))

            ei_c0 = snips_km[:, :, lab == 0].mean(axis=2).astype(np.float32)
            ei_c1 = snips_km[:, :, lab == 1].mean(axis=2).astype(np.float32)

            precheck = _timed_call(
                "step3d.kmeans_precheck_decision",
                kmeans_precheck_decision,
                vr,
                n0,
                n1,
                ei_c0,
                ei_c1,
                channel_tag=ch,
                timings=channel_stage_times,
                pc_var_thr=float(cfg["KM_PC_VAR_THR"]),
                minor_frac_thr=float(cfg["KM_MINOR_FRAC_THR"]),
                cos_oneunit_thr=float(cfg["KM_EI_COS_THR"]),
                asym_unique_ch_min=int(cfg["KM_ASYM_UNIQUE_CH_MIN"]),
                support_rel=float(cfg["SUPPORT_REL"]),
                support_abs=float(cfg["SUPPORT_ABS"]),
                cos_lag=int(cfg["KM_COS_LAG"]),
            )

            called_verdict = False
            shared_core = None
            shared_dirs = None
            m01 = None
            m10 = None

            if precheck["decided"]:
                verdict = precheck["verdict"]
                proceed = bool(precheck["proceed"])
                reason = precheck["reason"]
                detail = precheck["detail"]
            else:
                called_verdict = True
                verdict_info = _timed_call(
                    "step3e.verdict_from_kmeans",
                    verdict_from_kmeans,
                    ei_c0,
                    ei_c1,
                    channel_tag=ch,
                    timings=channel_stage_times,
                    max_lag=int(cfg["MAX_LAG"]),
                    support_rel=float(cfg["SUPPORT_REL"]),
                    support_abs=float(cfg["SUPPORT_ABS"]),
                    time_keep_rel=float(cfg["TIME_KEEP_REL"]),
                    frac_in_thr=float(cfg["FRAC_IN_THR"]),
                    out_in_ratio_thr=float(cfg["OUT_IN_RATIO_THR"]),
                    resid_frac_min=float(cfg["RESID_FRAC_MIN"]),
                    shared_cos_thr=float(cfg["SHARED_COS_THR"]),
                    shared_alpha_thr=float(cfg["SHARED_ALPHA_THR"]),
                )
                verdict = verdict_info["verdict"]
                proceed = bool(verdict_info["proceed"])
                reason = "verdict_from_kmeans"
                detail = precheck["detail"]
                shared_core = bool(verdict_info["shared_core"])
                shared_dirs = verdict_info["shared_dirs"]
                m01 = verdict_info["m01"]
                m10 = verdict_info["m10"]

            km_info = dict(
                n0=n0,
                n1=n1,
                ei_mean=ei_mean,
                sel=np.asarray(sel, dtype=np.int32),
                ei_c0=ei_c0,
                ei_c1=ei_c1,
                vr=np.asarray(vr, dtype=np.float32),
                verdict=verdict,
                proceed=bool(proceed),
                reason=reason,
                detail=detail,
                called_verdict=bool(called_verdict),
                precheck=precheck,
                shared_core=shared_core,
                shared_dirs=shared_dirs,
                m01=m01,
                m10=m10,
            )
            km_plot = dict(Xpc=Xpc[:, :2], labels=lab, vr=vr)

            audit["kmeans_verdict"] = str(km_info["verdict"])
            audit["kmeans_reason"] = str(km_info["reason"])

            if not km_info["proceed"]:
                _set_reject(audit, "kmeans_reject")
                return dict(status="rejected", audit=audit, km_plot=km_plot, km_info=km_info)

        if base_ei is None:
            left_times = np.asarray(step1["left_times"], dtype=np.int64)
            if left_times.size > int(cfg["SUBSAMPLE_MAX"]):
                pick = left_times[rng.choice(left_times.size, int(cfg["SUBSAMPLE_MAX"]), replace=False)]
            else:
                pick = left_times

            snips_base, _ = _timed_call(
                "step3a.extract_snippets_fast_ram[kmeans]",
                extract_snippets_fast_ram,
                raw_mod,
                pick,
                channel_tag=ch,
                timings=channel_stage_times,
                window=km_win,
                selected_channels=np.arange(C, dtype=np.int32),
            )
            audit["t_kmeans_extract_s"] = channel_stage_times.get(
                "step3a.extract_snippets_fast_ram[kmeans]", np.nan
            )
            if snips_base.shape[2] == 0:
                _set_reject(audit, "base_ei_empty")
                return dict(status="rejected", audit=audit)
            base_ei = snips_base.mean(axis=2).astype(np.float32)

        tr_candidate_times = np.sort(np.unique(np.asarray(step1["rightk_times_sorted"], dtype=np.int64)))

        support_ei_info = dict(
            main_ch=int(np.argmin(np.asarray(base_ei, dtype=np.float32).min(axis=1))),
            final_ei=np.asarray(base_ei, dtype=np.float32),
        )

        support_info = dict(ok=False, reason="not_run")
        clean_left_times = np.sort(np.unique(np.asarray(step1["left_times"], dtype=np.int64)))
        clean_right_times = np.sort(np.unique(np.asarray(tr_candidate_times, dtype=np.int64)))
        used_support_filter = False

        if bool(cfg["USE_BL_TR_SUPPORT_FILTER"]):
            support_info = run_bl_tr_support_for_channel(
                raw_mod,
                support_ei_info,
                step1["left_times"],
                tr_candidate_times,
                ch=ch,
                win=km_win,
                n_probe_per_side=int(cfg["SUPPORT_N_PROBE_PER_SIDE"]),
                n_top_channels=int(cfg["SUPPORT_TOP_CHANNELS"]),
                cos_mask_adc=float(cfg["SUPPORT_COS_MASK_ADC"]),
                k_peak=tuple(cfg["SUPPORT_K_PEAK"]),
                k_bulk=tuple(cfg["SUPPORT_K_BULK"]),
                min_bl_bulk=float(cfg["SUPPORT_MIN_BL_BULK"]),
                diag_eps=float(cfg["SUPPORT_DIAG_EPS"]),
            )

            if support_info.get("ok", False):
                used_support_filter = True
                bl_reject = np.asarray(support_info["bl_reject_times"], dtype=np.int64)
                bl_uncertain = np.asarray(support_info["bl_uncertain_times"], dtype=np.int64)
                if bl_reject.size or bl_uncertain.size:
                    bl_drop = np.sort(np.unique(np.concatenate([bl_reject, bl_uncertain])))
                else:
                    bl_drop = np.asarray([], dtype=np.int64)

                clean_left_times = np.setdiff1d(clean_left_times, bl_drop, assume_unique=False)
                clean_right_times = np.sort(np.unique(np.asarray(support_info["tr_keep_times"], dtype=np.int64)))

        final_times = np.sort(
            np.unique(
                np.concatenate([
                    np.asarray(clean_left_times, dtype=np.int64),
                    np.asarray(clean_right_times, dtype=np.int64),
                ])
            )
        )

        final_ei = np.asarray(base_ei, dtype=np.float32)
        main_ch_final = int(np.argmin(final_ei.min(axis=1)))
        audit["main_ch"] = int(main_ch_final)
        audit["final_times_n"] = int(final_times.size)
        audit["valley0_n"] = int(step1.get("valley_count", 0))
        audit["used_support_filter"] = bool(used_support_filter)

        if final_times.size < int(cfg["MIN_FINAL_SPIKES_TO_CALL_SUCCESS"]):
            _set_reject(audit, "too_few_final_spikes")
            return dict(status="rejected", audit=audit)

        support_summary = None
        uncertain_entry = None
        if support_info.get("ok", False):
            do = support_info["decision_out"]
            support_summary = dict(
                params=do["params"],
                bl_counts=do["bl_counts"],
                tr_counts=do["tr_counts"],
                bl_labels=np.asarray(do["bl_labels"], dtype=object),
                tr_labels=np.asarray(do["tr_labels"], dtype=object),
                bl_probe_times=np.asarray(support_info["probe"]["bl_times"], dtype=np.int64),
                tr_probe_times=np.asarray(support_info["probe"]["tr_times"], dtype=np.int64),
                bl_uncertain_times=np.asarray(support_info["bl_uncertain_times"], dtype=np.int64),
                tr_uncertain_times=np.asarray(support_info["tr_uncertain_times"], dtype=np.int64),
                tr_keep_times=np.asarray(support_info["tr_keep_times"], dtype=np.int64),
                bl_points=np.asarray([[m["BL_bulk"], m["TR_bulk"]] for m in do["bl_metrics"]], dtype=np.float32),
                tr_points=np.asarray([[m["BL_bulk"], m["TR_bulk"]] for m in do["tr_metrics"]], dtype=np.float32),
            )
            uncertain_entry = dict(
                detect_ch=int(ch),
                main_ch=int(main_ch_final),
                bl_uncertain_times=np.asarray(support_info["bl_uncertain_times"], dtype=np.int64),
                tr_uncertain_times=np.asarray(support_info["tr_uncertain_times"], dtype=np.int64),
                bl_probe_times=np.asarray(support_info["probe"]["bl_times"], dtype=np.int64),
                tr_probe_times=np.asarray(support_info["probe"]["tr_times"], dtype=np.int64),
            )

        final_proto = dict(
            main_ch=int(main_ch_final),
            final_ei=final_ei,
            final_times=final_times,
        )

        clean_bank = _timed_call(
            "step5.build_template_bank",
            build_template_bank,
            final_proto,
            channel_tag=ch,
            timings=channel_stage_times,
            raw_mod=raw_mod,
            win=km_win,
            sr=sr,
            n_bins=int(cfg["N_BINS"]),
            spikes_per=int(cfg["SPIKES_PER"]),
            ei_p2p_thr=float(cfg["EI_P2P_THR"]),
        )

        return dict(
            status="precomputed",
            audit=audit,
            step1=step1,
            tr_candidate_times=np.asarray(tr_candidate_times, dtype=np.int64),
            final_times=np.asarray(final_times, dtype=np.int64),
            final_ei=np.asarray(final_ei, dtype=np.float32),
            bank=clean_bank,
            kmeans=km_info,
            km_plot=km_plot,
            support_info=support_info,
            support_summary=support_summary,
            uncertain_entry=uncertain_entry,
            used_support_filter=bool(used_support_filter),
            isi_pairs_10_30=int(isi_pairs_10_30),
            timings=channel_stage_times,
        )

    except Exception as e:
        audit["status"] = "exception"
        audit["final_reason"] = "exception"
        audit["error_type"] = type(e).__name__
        audit["error_msg"] = str(e)
        return dict(status="exception", audit=audit)
    finally:
        audit["t_channel_total_s"] = time.perf_counter() - ch_t0
        if np.isnan(audit["t_km_window_s"]):
            audit["t_km_window_s"] = channel_stage_times.get("step2.choose_adaptive_km_window", np.nan)
        if np.isnan(audit["t_kmeans_extract_s"]):
            audit["t_kmeans_extract_s"] = channel_stage_times.get("step3a.extract_snippets_fast_ram[kmeans]", np.nan)
        if np.isnan(audit["t_step1_s"]):
            audit["t_step1_s"] = channel_stage_times.get("step1.find_valley_and_times", np.nan)


def main(exp_name, datafile_name):
    raw_dir = '/home/vyomr/Desktop/data/raw'
    binpath = os.path.join(ra.RAW_DIR, exp_name, datafile_name)
    rt = ra.RawTraces(binpath=binpath)
    rt.load_bin_data(verbose=True)

    ei_positions = ra.raw.D_ELECTRODE_MAPS['60um']
    # Following likes [Timepoints, Channels]
    raw_orig = rt.r_data.T
    print(f"raw_orig shape: {raw_orig.shape}")

    os.makedirs(os.path.join(raw_dir, exp_name), exist_ok=True)
    baseline_path = os.path.join(raw_dir, exp_name, f"{datafile_name}_baseline_derivative_20k.json")

    segment_len = 20_000
    if os.path.exists(baseline_path):
        print(f"Loading baselines")
        with open(baseline_path, 'r') as f:
            data = json.load(f)
        baselines = np.array(data['baselines'], dtype=np.float32)
    else:
        print(f"Computing baselines")
        baselines = axolotl_utils_ram.compute_baselines_int16_deriv_robust(raw_orig, segment_len=segment_len, diff_thresh=10, trim_fraction=0.15) # shape (512, 360)

        with open(baseline_path, 'w') as f:
            json.dump({
                'baselines': baselines.tolist(),
            }, f)

    print("subtracting baselines")

    axolotl_utils_ram.subtract_segment_baselines_int16(raw_data=raw_orig,
                                        baselines_f32=baselines,
                                        segment_len=segment_len) 

    raw_mod = raw_orig
    del raw_orig   # optional

    # ============================================================
    # LH loop with per-channel audit trail + end-of-run summaries
    # ============================================================

    # assert "raw_mod" in globals(), "Need raw_mod (baseline-subtracted int16) loaded."
    # assert "ei_positions" in globals(), "Need ei_positions loaded."
    # assert "extract_snippets_fast_ram" in globals(), "Need extract_snippets_fast_ram imported."
    # assert "find_valley_and_times" in globals(), "Need find_valley_and_times imported."

    T, C = raw_mod.shape
    sr = float(globals().get("sample_rate_hz", 20_000))

    # ----------------------------
    # USER KNOBS
    # ----------------------------
    CHANNELS_TO_RUN = list(range(C))   # debug: [103], or all channels: list(range(C))
    PLOT_DIAGNOSTICS = True            # per-success plots during the run
    PLOT_END_SUMMARY = True            # timing/window summary plots after the run
    RESET_RESULTS = True
    VERBOSE_TIMING = False
    PARALLEL_CHANNEL_PRECOMPUTE = True
    PARALLEL_WORKERS = max(1, (os.cpu_count() or 1) - 1)

    win = (-20, 40)

    # Step 1 valley
    BIN_WIDTH = 10.0
    VALLEY_BINS = 5
    MIN_VALID_COUNT = 900
    RATIO_BASE = 3
    RATIO_STEP = 500
    RATIO_FLOOR = 2
    RATIO_CAP = 10
    MAX_VALLEY_COUNT = 500

    # Early reject checks
    ISI_10_30_MAX = 10

    # Adaptive EI / k-means window
    KM_PROBE_N = 500
    KM_PROBE_WIN = (-40, 80)
    KM_PROBE_TIME_AMP_THR = 30.0
    KM_PROBE_CH_P2P_THR = 30.0
    KM_WIN_PAD_LEFT = 3
    KM_WIN_PAD_RIGHT = 3
    KM_WIN_MIN_PRE = 15
    KM_WIN_MIN_POST = 30

    # Left-only KMeans / verdict
    DO_KMEANS = True
    SUBSAMPLE_MAX = 5_000
    RMS_THRESH = 5.0
    N_PC = 2
    RNG = np.random.RandomState(123)

    KM_PC_VAR_THR = 0.10
    KM_MINOR_FRAC_THR = 0.10
    KM_EI_COS_THR = 0.95
    KM_ASYM_UNIQUE_CH_MIN = 3
    KM_COS_LAG = 1

    MAX_LAG = 3
    SUPPORT_REL = 0.10
    SUPPORT_ABS = 30.0
    TIME_KEEP_REL = 0.10
    FRAC_IN_THR = 0.20
    OUT_IN_RATIO_THR = 2.0
    RESID_FRAC_MIN = 0.08
    SHARED_COS_THR = 0.95
    SHARED_ALPHA_THR = 0.95

    # Subtraction bank / subtraction
    N_BINS = 5
    SPIKES_PER = 5000
    EI_P2P_THR = 50.0

    TIME_KEEP_FRAC = 0.08
    WEIGHT_POWER = 1.0

    JMAX_SUB = 3
    BATCH_SUB = 256
    USE_CH_WEIGHT_SUB = True
    MIN_SEP_MS = 1.0
    COS_THR_SUB = 0.90

    MIN_FINAL_SPIKES_TO_CALL_SUCCESS = 200

    # BL/TR support filter
    USE_BL_TR_SUPPORT_FILTER = True
    SUPPORT_N_PROBE_PER_SIDE = 2000
    SUPPORT_TOP_CHANNELS = 12
    SUPPORT_COS_MASK_ADC = 30.0
    SUPPORT_K_PEAK = (5, 10, 20)
    SUPPORT_K_BULK = (50, 100, 200)
    SUPPORT_MIN_BL_BULK = 0.70
    SUPPORT_DIAG_EPS = 0.05


    # ----------------------------
    # Reset / persistent outputs
    # ----------------------------
    if RESET_RESULTS or ("lh_units" not in globals()):
        lh_units = []
    if RESET_RESULTS or ("lh_uncertain_ledger" not in globals()):
        lh_uncertain_ledger = []
    if RESET_RESULTS or ("lh_channel_audit" not in globals()):
        lh_channel_audit = []

    # fresh run-local summaries
    skip_counts = defaultdict(int)


    t_loop0 = time.time()
    print(f"Starting LH loop over {len(CHANNELS_TO_RUN)} channels...")

    channels_to_process = CHANNELS_TO_RUN

    if PARALLEL_CHANNEL_PRECOMPUTE and len(CHANNELS_TO_RUN) > 0:
        global _PARALLEL_RAW_MOD, _PARALLEL_CFG

        _PARALLEL_RAW_MOD = raw_mod
        _PARALLEL_CFG = dict(
            C=int(C),
            sr=float(sr),
            win=tuple(win),
            RNG_SEED=123,
            BIN_WIDTH=BIN_WIDTH,
            VALLEY_BINS=VALLEY_BINS,
            MIN_VALID_COUNT=MIN_VALID_COUNT,
            RATIO_BASE=RATIO_BASE,
            RATIO_STEP=RATIO_STEP,
            RATIO_FLOOR=RATIO_FLOOR,
            RATIO_CAP=RATIO_CAP,
            MAX_VALLEY_COUNT=MAX_VALLEY_COUNT,
            ISI_10_30_MAX=ISI_10_30_MAX,
            KM_PROBE_N=KM_PROBE_N,
            KM_PROBE_WIN=KM_PROBE_WIN,
            KM_PROBE_TIME_AMP_THR=KM_PROBE_TIME_AMP_THR,
            KM_PROBE_CH_P2P_THR=KM_PROBE_CH_P2P_THR,
            KM_WIN_PAD_LEFT=KM_WIN_PAD_LEFT,
            KM_WIN_PAD_RIGHT=KM_WIN_PAD_RIGHT,
            KM_WIN_MIN_PRE=KM_WIN_MIN_PRE,
            KM_WIN_MIN_POST=KM_WIN_MIN_POST,
            DO_KMEANS=DO_KMEANS,
            SUBSAMPLE_MAX=SUBSAMPLE_MAX,
            RMS_THRESH=RMS_THRESH,
            N_PC=N_PC,
            KM_PC_VAR_THR=KM_PC_VAR_THR,
            KM_MINOR_FRAC_THR=KM_MINOR_FRAC_THR,
            KM_EI_COS_THR=KM_EI_COS_THR,
            KM_ASYM_UNIQUE_CH_MIN=KM_ASYM_UNIQUE_CH_MIN,
            KM_COS_LAG=KM_COS_LAG,
            MAX_LAG=MAX_LAG,
            SUPPORT_REL=SUPPORT_REL,
            SUPPORT_ABS=SUPPORT_ABS,
            TIME_KEEP_REL=TIME_KEEP_REL,
            FRAC_IN_THR=FRAC_IN_THR,
            OUT_IN_RATIO_THR=OUT_IN_RATIO_THR,
            RESID_FRAC_MIN=RESID_FRAC_MIN,
            SHARED_COS_THR=SHARED_COS_THR,
            SHARED_ALPHA_THR=SHARED_ALPHA_THR,
            MIN_FINAL_SPIKES_TO_CALL_SUCCESS=MIN_FINAL_SPIKES_TO_CALL_SUCCESS,
            USE_BL_TR_SUPPORT_FILTER=USE_BL_TR_SUPPORT_FILTER,
            SUPPORT_N_PROBE_PER_SIDE=SUPPORT_N_PROBE_PER_SIDE,
            SUPPORT_TOP_CHANNELS=SUPPORT_TOP_CHANNELS,
            SUPPORT_COS_MASK_ADC=SUPPORT_COS_MASK_ADC,
            SUPPORT_K_PEAK=SUPPORT_K_PEAK,
            SUPPORT_K_BULK=SUPPORT_K_BULK,
            SUPPORT_MIN_BL_BULK=SUPPORT_MIN_BL_BULK,
            SUPPORT_DIAG_EPS=SUPPORT_DIAG_EPS,
            N_BINS=N_BINS,
            SPIKES_PER=SPIKES_PER,
            EI_P2P_THR=EI_P2P_THR,
        )

        precompute_by_ch = {}
        max_workers = min(int(PARALLEL_WORKERS), len(CHANNELS_TO_RUN))
        print(f"Parallel precompute enabled: workers={max_workers}")

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("fork")) as ex:
            future_to_ch = {
                ex.submit(_channel_precompute_worker, int(ch)): int(ch)
                for ch in CHANNELS_TO_RUN
            }

            for fut in as_completed(future_to_ch):
                ch = future_to_ch[fut]
                try:
                    precompute_by_ch[ch] = fut.result()
                except Exception as e:
                    audit = _new_audit_record(ch)
                    audit["status"] = "exception"
                    audit["final_reason"] = "exception"
                    audit["error_type"] = type(e).__name__
                    audit["error_msg"] = str(e)
                    precompute_by_ch[ch] = dict(status="exception", audit=audit)

        _PARALLEL_RAW_MOD = None
        _PARALLEL_CFG = None
        channels_to_process = []

        for ch in CHANNELS_TO_RUN:
            rec = precompute_by_ch.get(int(ch), None)
            if rec is None:
                audit = _new_audit_record(ch)
                audit["status"] = "exception"
                audit["final_reason"] = "exception"
                audit["error_type"] = "RuntimeError"
                audit["error_msg"] = "Missing precompute result"
                lh_channel_audit.append(audit)
                skip_counts["exception"] += 1
                continue

            audit = rec["audit"]
            status = rec.get("status", "exception")

            if status != "precomputed":
                if status == "rejected" and audit.get("final_reason") == "kmeans_reject":
                    km_plot = rec.get("km_plot", None)
                    km_info = rec.get("km_info", None)
                    if PLOT_DIAGNOSTICS and km_plot is not None and km_info is not None:
                        plot_kmeans_pc_scatter(
                            km_plot,
                            title=f"CH {ch} | {km_info['verdict']} | {km_info['reason']}"
                        )

                reason = str(audit.get("final_reason", "unknown"))
                skip_counts[reason] += 1
                lh_channel_audit.append(audit)
                continue

            channel_stage_times = {}
            clean_bank = rec["bank"]
            final_times = np.asarray(rec["final_times"], dtype=np.int64)

            match = _timed_call(
                "step6.subtract_lh_masked",
                subtract_lh_masked,
                raw_mod,
                final_times,
                clean_bank,
                channel_tag=ch,
                timings=channel_stage_times,
                time_keep_frac=TIME_KEEP_FRAC,
                weight_power=WEIGHT_POWER,
                use_ch_weight_sub=USE_CH_WEIGHT_SUB,
                min_sep_ms=MIN_SEP_MS,
                cos_thr_sub=COS_THR_SUB,
                jmax_sub=JMAX_SUB,
                batch_sub=BATCH_SUB,
            )

            if match is None or int(match.get("n_sub", 0)) == 0:
                _set_reject(audit, "subtraction_empty")
                skip_counts["subtraction_empty"] += 1
                lh_channel_audit.append(audit)
                continue

            final_spikes_main = np.asarray(match["spike_times_main"], dtype=np.int64)
            audit["final_spikes_main_n"] = int(final_spikes_main.size)

            unit = dict(
                detect_ch=int(ch),
                step1=rec["step1"],
                tr_candidate_times=np.asarray(rec["tr_candidate_times"], dtype=np.int64),
                final_times=np.asarray(rec["final_times"], dtype=np.int64),
                final_ei=np.asarray(rec["final_ei"], dtype=np.float32),
                bank=clean_bank,
                match=match,
                final_spikes_main=final_spikes_main,
                kmeans=rec.get("kmeans", None),
                support=rec.get("support_summary", None),
            )
            unit["diag_isi_pairs_10_30"] = int(rec["isi_pairs_10_30"])
            unit["used_support_filter"] = bool(rec.get("used_support_filter", False))

            lh_units.append(unit)
            audit["unit_index"] = int(len(lh_units))
            _set_success(audit)
            skip_counts["SUCCESS"] += 1

            support_info = rec.get("support_info", dict(ok=False))
            if support_info.get("ok", False):
                do = support_info["decision_out"]
                blc = do["bl_counts"]
                trc = do["tr_counts"]

                bl_probe_soup_n = int(blc.get("soup", 0))
                bl_probe_unc_n = int(blc.get("uncertain_boundary", 0)) + int(blc.get("uncertain_lowBL", 0))
                tr_probe_lh_n = int(trc.get("LH", 0))

                audit["bl_probe_to_soup"] = bl_probe_soup_n
                audit["bl_probe_to_unc"] = bl_probe_unc_n
                audit["tr_probe_to_lh"] = tr_probe_lh_n

                extra_msg = (
                    f" | valley0={audit['valley0_n']}"
                    f" | BLprobe→soup={bl_probe_soup_n}"
                    f" | BLprobe→unc={bl_probe_unc_n}"
                    f" | TRprobe→LH={tr_probe_lh_n}"
                )
            else:
                extra_msg = f" | valley0={audit['valley0_n']} | BL/TRprobe=NA"

            print(
                f"✅ SUCCESS unit#{len(lh_units)} | detect_ch={ch} | "
                f"main_ch={match['main_ch']} | Nspikes={final_spikes_main.size}"
                f"{extra_msg}"
            )

            uncertain_entry = rec.get("uncertain_entry", None)
            if uncertain_entry is not None:
                lh_uncertain_ledger.append(uncertain_entry)

            if PLOT_DIAGNOSTICS:
                _t_stage = _stage_start("step7.plot_diagnostics", channel_tag=ch)

                plot_amp_hist(rec["step1"], ch)
                km_plot = rec.get("km_plot", None)
                km_info = rec.get("kmeans", None)
                if km_plot is not None and km_info is not None:
                    plot_kmeans_pc_scatter(
                        km_plot,
                        title=f"CH {ch} | {km_info['verdict']} | {km_info['reason']}"
                    )
                if support_info.get("ok", False):
                    plot_bl_tr_support_scatter(
                        support_info["decision_out"],
                        title=f"CH {ch} | BL/TR bulk support",
                    )
                plot_final_ei(
                    np.asarray(rec["final_ei"], dtype=np.float32),
                    clean_bank["main_ch"],
                    title=f"Final EI | detect_ch={ch} main_ch={clean_bank['main_ch']} | N={final_spikes_main.size}",
                    ei_positions=ei_positions
                )

                _stage_end("step7.plot_diagnostics", _t_stage, timings=channel_stage_times, channel_tag=ch)

            lh_channel_audit.append(audit)

            try:
                plt.close("all")
            except Exception:
                pass

    for ch in channels_to_process:
        channel_stage_times = {}
        ch_t0 = time.perf_counter()
        audit = _new_audit_record(ch)

        try:
            # ----------------------------
            # STEP 1: valley detection
            # ----------------------------
            step1 = _timed_call(
                "step1.find_valley_and_times",
                find_valley_and_times,
                raw_mod,
                ch,
                channel_tag=ch,
                timings=channel_stage_times,
                window=win,
                start=0,
                stop=None,
                bin_width=BIN_WIDTH,
                valley_bins=VALLEY_BINS,
                min_valid_count=MIN_VALID_COUNT,
                ratio_base=RATIO_BASE,
                ratio_step=RATIO_STEP,
                ratio_floor=RATIO_FLOOR,
                ratio_cap=RATIO_CAP,
                right_k=SUPPORT_N_PROBE_PER_SIDE,
                min_trough=-2500,
            )

            audit["step1_accepted"] = bool(step1.get("accepted", False))
            audit["left_count"] = int(step1.get("left_count", 0))
            audit["valley_count"] = int(step1.get("valley_count", 0))
            audit["required_ratio"] = (
                None if step1.get("required_ratio", None) is None else float(step1.get("required_ratio"))
            )
            audit["valley_low"] = step1.get("valley_low", None)
            audit["valley_high"] = step1.get("valley_high", None)
            audit["t_step1_s"] = channel_stage_times.get("step1.find_valley_and_times", np.nan)

            if int(step1.get("valley_count", 0)) > int(MAX_VALLEY_COUNT):
                skip_counts["valley_count>max"] += 1
                _set_reject(audit, "valley_count>max")
                print(
                    f"[CH {ch}] REJECT valley_count>max | "
                    f"valley_count={int(step1.get('valley_count', 0))}"
                )
                continue

            if not step1.get("accepted", False):
                skip_counts["valley_not_accepted"] += 1
                _set_reject(audit, "valley_not_accepted")
                print(
                    f"[CH {ch}] REJECT valley_not_accepted | "
                    f"left={int(step1.get('left_count', -1))} "
                    f"valley={int(step1.get('valley_count', -1))}"
                )
                continue

            left_times = np.asarray(step1["left_times"], dtype=np.int64)

            km_win, km_win_info = _timed_call(
                "step2.choose_adaptive_km_window",
                choose_adaptive_km_window,
                raw_mod,
                left_times,
                channel_tag=ch,
                timings=channel_stage_times,
                probe_n=KM_PROBE_N,
                probe_win=KM_PROBE_WIN,
                fallback_win=win,
                time_amp_thr=KM_PROBE_TIME_AMP_THR,
                ch_ptp_thr=KM_PROBE_CH_P2P_THR,
                pad_left=KM_WIN_PAD_LEFT,
                pad_right=KM_WIN_PAD_RIGHT,
                min_pre=KM_WIN_MIN_PRE,
                min_post=KM_WIN_MIN_POST,
                rng=RNG,
            )

            audit["km_win_pre"] = int(km_win[0])
            audit["km_win_post"] = int(km_win[1])
            audit["km_probe_status"] = str(km_win_info.get("status", "unknown"))
            audit["km_probe_left_rel"] = km_win_info.get("left_rel", None)
            audit["km_probe_right_rel"] = km_win_info.get("right_rel", None)
            audit["km_probe_n_req"] = km_win_info.get("probe_n_req", None)
            audit["km_probe_n_valid"] = km_win_info.get("probe_n_valid", None)
            audit["km_probe_n_ch"] = km_win_info.get("n_ch_keep", None)
            audit["t_km_window_s"] = channel_stage_times.get("step2.choose_adaptive_km_window", np.nan)

            print(
                f"[CH {ch}] km_win={km_win} | "
                f"probe={km_win_info['status']} | "
                f"span={km_win_info['left_rel']}..{km_win_info['right_rel']} | "
                f"nprobe={km_win_info['probe_n_valid']}/{km_win_info['probe_n_req']} | "
                f"nch={km_win_info['n_ch_keep']}"
            )

            # ----------------------------
            # Early reject checks
            # ----------------------------
            isi_pairs_10_30 = _timed_call(
                "step2.compute_left_isi_pairs_10_30",
                compute_left_isi_pairs_10_30,
                step1,
                channel_tag=ch,
                timings=channel_stage_times,
            )
            audit["isi_pairs_10_30"] = int(isi_pairs_10_30)

            if isi_pairs_10_30 > ISI_10_30_MAX:
                skip_counts["abort_ISI10_30"] += 1
                _set_reject(audit, "abort_ISI10_30")
                print(f"[CH {ch}] REJECT abort_ISI10_30 | pairs={isi_pairs_10_30}")
                continue

            # ----------------------------
            # KMeans verdict on LEFT spikes
            # ----------------------------
            km_info = None
            km_plot = None
            base_ei = None

            if DO_KMEANS:
                left_times = np.asarray(step1["left_times"], dtype=np.int64)

                if left_times.size > int(SUBSAMPLE_MAX):
                    pick = left_times[RNG.choice(left_times.size, int(SUBSAMPLE_MAX), replace=False)]
                else:
                    pick = left_times

                snips_km, valid_times_km = _timed_call(
                    "step3a.extract_snippets_fast_ram[kmeans]",
                    extract_snippets_fast_ram,
                    raw_mod,
                    pick,
                    channel_tag=ch,
                    timings=channel_stage_times,
                    window=km_win,
                    selected_channels=np.arange(C, dtype=np.int32),
                )
                audit["t_kmeans_extract_s"] = channel_stage_times.get(
                    "step3a.extract_snippets_fast_ram[kmeans]", np.nan
                )

                N = snips_km.shape[2]

                ei_mean = snips_km.mean(axis=2).astype(np.float32)
                base_ei = ei_mean
                rms = np.sqrt(np.mean(ei_mean ** 2, axis=1))
                sel = np.flatnonzero(rms > RMS_THRESH)
                if sel.size == 0:
                    sel = np.argsort(rms)[-16:]
                    sel.sort()

                X = snips_km[sel, :, :].transpose(2, 0, 1).reshape(
                    N, sel.size * snips_km.shape[1]
                ).astype(np.float32)

                n_pc = int(min(N_PC, X.shape[0], X.shape[1]))
                pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0)
                Xpc = _timed_call(
                    "step3b.PCA.fit_transform",
                    pca.fit_transform,
                    X,
                    channel_tag=ch,
                    timings=channel_stage_times,
                )
                vr = pca.explained_variance_ratio_

                km = KMeans(n_clusters=2, n_init=50, random_state=0)
                lab = _timed_call(
                    "step3c.KMeans.fit_predict",
                    km.fit_predict,
                    Xpc,
                    channel_tag=ch,
                    timings=channel_stage_times,
                )

                n0 = int(np.sum(lab == 0))
                n1 = int(np.sum(lab == 1))

                ei_c0 = snips_km[:, :, lab == 0].mean(axis=2).astype(np.float32)
                ei_c1 = snips_km[:, :, lab == 1].mean(axis=2).astype(np.float32)

                precheck = _timed_call(
                    "step3d.kmeans_precheck_decision",
                    kmeans_precheck_decision,
                    vr,
                    n0,
                    n1,
                    ei_c0,
                    ei_c1,
                    channel_tag=ch,
                    timings=channel_stage_times,
                    pc_var_thr=KM_PC_VAR_THR,
                    minor_frac_thr=KM_MINOR_FRAC_THR,
                    cos_oneunit_thr=KM_EI_COS_THR,
                    asym_unique_ch_min=KM_ASYM_UNIQUE_CH_MIN,
                    support_rel=SUPPORT_REL,
                    support_abs=SUPPORT_ABS,
                    cos_lag=KM_COS_LAG,
                )

                called_verdict = False
                shared_core = None
                shared_dirs = None
                m01 = None
                m10 = None

                if precheck["decided"]:
                    verdict = precheck["verdict"]
                    proceed = bool(precheck["proceed"])
                    reason = precheck["reason"]
                    detail = precheck["detail"]
                else:
                    called_verdict = True
                    verdict_info = _timed_call(
                        "step3e.verdict_from_kmeans",
                        verdict_from_kmeans,
                        ei_c0,
                        ei_c1,
                        channel_tag=ch,
                        timings=channel_stage_times,
                        max_lag=MAX_LAG,
                        support_rel=SUPPORT_REL,
                        support_abs=SUPPORT_ABS,
                        time_keep_rel=TIME_KEEP_REL,
                        frac_in_thr=FRAC_IN_THR,
                        out_in_ratio_thr=OUT_IN_RATIO_THR,
                        resid_frac_min=RESID_FRAC_MIN,
                        shared_cos_thr=SHARED_COS_THR,
                        shared_alpha_thr=SHARED_ALPHA_THR,
                    )
                    verdict = verdict_info["verdict"]
                    proceed = bool(verdict_info["proceed"])
                    reason = "verdict_from_kmeans"
                    detail = precheck["detail"]
                    shared_core = bool(verdict_info["shared_core"])
                    shared_dirs = verdict_info["shared_dirs"]
                    m01 = verdict_info["m01"]
                    m10 = verdict_info["m10"]

                km_info = dict(
                    n0=n0,
                    n1=n1,
                    ei_mean=ei_mean,
                    sel=np.asarray(sel, dtype=np.int32),
                    ei_c0=ei_c0,
                    ei_c1=ei_c1,
                    vr=np.asarray(vr, dtype=np.float32),
                    verdict=verdict,
                    proceed=bool(proceed),
                    reason=reason,
                    detail=detail,
                    called_verdict=bool(called_verdict),
                    precheck=precheck,
                    shared_core=shared_core,
                    shared_dirs=shared_dirs,
                    m01=m01,
                    m10=m10,
                )
                km_plot = dict(Xpc=Xpc[:, :2], labels=lab, vr=vr)

                audit["kmeans_verdict"] = str(km_info["verdict"])
                audit["kmeans_reason"] = str(km_info["reason"])

                print(
                    f"[CH {ch}] KMEANS | verdict={km_info['verdict']} | "
                    f"reason={km_info['reason']} | detail={km_info['detail']} | "
                    f"called_verdict={km_info['called_verdict']}"
                )

                if not km_info["proceed"]:
                    skip_counts["kmeans_reject"] += 1
                    _set_reject(audit, "kmeans_reject")
                    print(f"[CH {ch}] REJECT kmeans_reject | verdict={km_info['verdict']}")

                    if PLOT_DIAGNOSTICS and km_plot is not None:
                        plot_kmeans_pc_scatter(
                            km_plot,
                            title=f"CH {ch} | {km_info['verdict']} | {km_info['reason']}"
                        )

                    continue

            if base_ei is None:
                left_times = np.asarray(step1["left_times"], dtype=np.int64)

                if left_times.size > int(SUBSAMPLE_MAX):
                    pick = left_times[RNG.choice(left_times.size, int(SUBSAMPLE_MAX), replace=False)]
                else:
                    pick = left_times

                snips_base, valid_times_base = _timed_call(
                    "step3a.extract_snippets_fast_ram[kmeans]",
                    extract_snippets_fast_ram,
                    raw_mod,
                    pick,
                    channel_tag=ch,
                    timings=channel_stage_times,
                    window=km_win,
                    selected_channels=np.arange(C, dtype=np.int32),
                )
                audit["t_kmeans_extract_s"] = channel_stage_times.get(
                    "step3a.extract_snippets_fast_ram[kmeans]", np.nan
                )

                if snips_base.shape[2] == 0:
                    skip_counts["base_ei_empty"] += 1
                    _set_reject(audit, "base_ei_empty")
                    print(f"[CH {ch}] REJECT base_ei_empty")
                    continue

                base_ei = snips_base.mean(axis=2).astype(np.float32)

                print(
                    f"[CH {ch}] KMEANS skipped | using mean EI from LEFT spikes "
                    f"(N={snips_base.shape[2]}) | km_win={km_win}"
                )

            # ----------------------------
            # BL/TR candidates direct from step 1
            # ----------------------------
            tr_candidate_times = np.sort(
                np.unique(np.asarray(step1["rightk_times_sorted"], dtype=np.int64))
            )

            support_ei_info = dict(
                main_ch=int(np.argmin(np.asarray(base_ei, dtype=np.float32).min(axis=1))),
                final_ei=np.asarray(base_ei, dtype=np.float32),
            )

            # ----------------------------
            # BL/TR support filter
            # ----------------------------
            support_info = dict(ok=False, reason="not_run")
            clean_left_times = np.sort(np.unique(np.asarray(step1["left_times"], dtype=np.int64)))
            clean_right_times = np.sort(np.unique(np.asarray(tr_candidate_times, dtype=np.int64)))
            used_support_filter = False

            if USE_BL_TR_SUPPORT_FILTER:
                support_info = run_bl_tr_support_for_channel(
                    raw_mod,
                    support_ei_info,
                    step1["left_times"],
                    tr_candidate_times,
                    ch=ch,
                    win=km_win,
                    n_probe_per_side=SUPPORT_N_PROBE_PER_SIDE,
                    n_top_channels=SUPPORT_TOP_CHANNELS,
                    cos_mask_adc=SUPPORT_COS_MASK_ADC,
                    k_peak=SUPPORT_K_PEAK,
                    k_bulk=SUPPORT_K_BULK,
                    min_bl_bulk=SUPPORT_MIN_BL_BULK,
                    diag_eps=SUPPORT_DIAG_EPS,
                )

                if support_info.get("ok", False):
                    used_support_filter = True
                    bl_reject = np.asarray(support_info["bl_reject_times"], dtype=np.int64)
                    bl_uncertain = np.asarray(support_info["bl_uncertain_times"], dtype=np.int64)

                    if bl_reject.size or bl_uncertain.size:
                        bl_drop = np.sort(np.unique(np.concatenate([bl_reject, bl_uncertain])))
                    else:
                        bl_drop = np.asarray([], dtype=np.int64)

                    clean_left_times = np.setdiff1d(clean_left_times, bl_drop, assume_unique=False)
                    clean_right_times = np.sort(
                        np.unique(np.asarray(support_info["tr_keep_times"], dtype=np.int64))
                    )

            # ----------------------------
            # Final spike list + final EI
            # ----------------------------
            final_times = np.sort(
                np.unique(
                    np.concatenate([
                        np.asarray(clean_left_times, dtype=np.int64),
                        np.asarray(clean_right_times, dtype=np.int64),
                    ])
                )
            )

            final_ei = np.asarray(base_ei, dtype=np.float32)
            main_ch_final = int(np.argmin(final_ei.min(axis=1)))
            audit["main_ch"] = int(main_ch_final)
            audit["final_times_n"] = int(final_times.size)
            audit["valley0_n"] = int(step1.get("valley_count", 0))
            audit["used_support_filter"] = bool(used_support_filter)

            if main_ch_final != int(ch):
                print(
                    f"[CH {ch}] NOTE main/ref mismatch | "
                    f"ref_ch={int(ch)} main_ch={main_ch_final}"
                )

            if final_times.size < int(MIN_FINAL_SPIKES_TO_CALL_SUCCESS):
                skip_counts["too_few_final_spikes"] += 1
                _set_reject(audit, "too_few_final_spikes")
                print(f"[CH {ch}] REJECT too_few_final_spikes | n={final_times.size}")
                continue

            support_summary = None
            if support_info.get("ok", False):
                do = support_info["decision_out"]
                support_summary = dict(
                    params=do["params"],
                    bl_counts=do["bl_counts"],
                    tr_counts=do["tr_counts"],
                    bl_labels=np.asarray(do["bl_labels"], dtype=object),
                    tr_labels=np.asarray(do["tr_labels"], dtype=object),
                    bl_probe_times=np.asarray(support_info["probe"]["bl_times"], dtype=np.int64),
                    tr_probe_times=np.asarray(support_info["probe"]["tr_times"], dtype=np.int64),
                    bl_uncertain_times=np.asarray(support_info["bl_uncertain_times"], dtype=np.int64),
                    tr_uncertain_times=np.asarray(support_info["tr_uncertain_times"], dtype=np.int64),
                    tr_keep_times=np.asarray(support_info["tr_keep_times"], dtype=np.int64),
                    bl_points=np.asarray(
                        [[m["BL_bulk"], m["TR_bulk"]] for m in do["bl_metrics"]],
                        dtype=np.float32,
                    ),
                    tr_points=np.asarray(
                        [[m["BL_bulk"], m["TR_bulk"]] for m in do["tr_metrics"]],
                        dtype=np.float32,
                    ),
                )

                lh_uncertain_ledger.append(
                    dict(
                        detect_ch=int(ch),
                        main_ch=int(main_ch_final),
                        bl_uncertain_times=np.asarray(support_info["bl_uncertain_times"], dtype=np.int64),
                        tr_uncertain_times=np.asarray(support_info["tr_uncertain_times"], dtype=np.int64),
                        bl_probe_times=np.asarray(support_info["probe"]["bl_times"], dtype=np.int64),
                        tr_probe_times=np.asarray(support_info["probe"]["tr_times"], dtype=np.int64),
                    )
                )

            # ----------------------------
            # Commit stage: only now build subtraction bank and subtract
            # ----------------------------
            final_proto = dict(
                main_ch=int(main_ch_final),
                final_ei=final_ei,
                final_times=final_times,
            )

            clean_bank = _timed_call(
                "step5.build_template_bank",
                build_template_bank,
                final_proto,
                channel_tag=ch,
                timings=channel_stage_times,
                raw_mod=raw_mod,
                win=km_win,
                sr=sr,
                n_bins=N_BINS,
                spikes_per=SPIKES_PER,
                ei_p2p_thr=EI_P2P_THR,
            )

            spike_times_to_subtract = np.asarray(final_times, dtype=np.int64)

            match = _timed_call(
                "step6.subtract_lh_masked",
                subtract_lh_masked,
                raw_mod,
                spike_times_to_subtract,
                clean_bank,
                channel_tag=ch,
                timings=channel_stage_times,
                time_keep_frac=TIME_KEEP_FRAC,
                weight_power=WEIGHT_POWER,
                use_ch_weight_sub=USE_CH_WEIGHT_SUB,
                min_sep_ms=MIN_SEP_MS,
                cos_thr_sub=COS_THR_SUB,
                jmax_sub=JMAX_SUB,
                batch_sub=BATCH_SUB,
            )

            if match is None or int(match.get("n_sub", 0)) == 0:
                skip_counts["subtraction_empty"] += 1
                _set_reject(audit, "subtraction_empty")
                print(f"[CH {ch}] REJECT subtraction_empty")
                continue

            final_spikes_main = np.asarray(match["spike_times_main"], dtype=np.int64)
            audit["final_spikes_main_n"] = int(final_spikes_main.size)

            unit = dict(
                detect_ch=int(ch),
                step1=step1,
                tr_candidate_times=np.asarray(tr_candidate_times, dtype=np.int64),
                final_times=np.asarray(final_times, dtype=np.int64),
                final_ei=np.asarray(final_ei, dtype=np.float32),
                bank=clean_bank,
                match=match,
                final_spikes_main=final_spikes_main,
                kmeans=km_info,
                support=support_summary,
            )

            unit["diag_isi_pairs_10_30"] = int(isi_pairs_10_30)
            unit["used_support_filter"] = bool(used_support_filter)

            lh_units.append(unit)
            audit["unit_index"] = int(len(lh_units))
            _set_success(audit)
            skip_counts["SUCCESS"] += 1

            if support_info.get("ok", False):
                do = support_info["decision_out"]
                blc = do["bl_counts"]
                trc = do["tr_counts"]

                bl_probe_soup_n = int(blc.get("soup", 0))
                bl_probe_unc_n = int(blc.get("uncertain_boundary", 0)) + int(blc.get("uncertain_lowBL", 0))
                tr_probe_lh_n = int(trc.get("LH", 0))

                audit["bl_probe_to_soup"] = bl_probe_soup_n
                audit["bl_probe_to_unc"] = bl_probe_unc_n
                audit["tr_probe_to_lh"] = tr_probe_lh_n

                extra_msg = (
                    f" | valley0={audit['valley0_n']}"
                    f" | BLprobe→soup={bl_probe_soup_n}"
                    f" | BLprobe→unc={bl_probe_unc_n}"
                    f" | TRprobe→LH={tr_probe_lh_n}"
                )
            else:
                extra_msg = f" | valley0={audit['valley0_n']} | BL/TRprobe=NA"

            print(
                f"✅ SUCCESS unit#{len(lh_units)} | detect_ch={ch} | "
                f"main_ch={match['main_ch']} | Nspikes={final_spikes_main.size}"
                f"{extra_msg}"
            )

            if PLOT_DIAGNOSTICS:
                _t_stage = _stage_start("step7.plot_diagnostics", channel_tag=ch)

                plot_amp_hist(step1, ch)
                if km_plot is not None:
                    plot_kmeans_pc_scatter(
                        km_plot,
                        title=f"CH {ch} | {km_info['verdict']} | {km_info['reason']}"
                    )
                if support_info.get("ok", False):
                    plot_bl_tr_support_scatter(
                        support_info["decision_out"],
                        title=f"CH {ch} | BL/TR bulk support",
                    )
                plot_final_ei(
                    final_ei,
                    clean_bank["main_ch"],
                    title=f"Final EI | detect_ch={ch} main_ch={clean_bank['main_ch']} | N={final_spikes_main.size}",
                    ei_positions=ei_positions
                )

                _stage_end("step7.plot_diagnostics", _t_stage, timings=channel_stage_times, channel_tag=ch)

        except Exception as e:
            skip_counts["exception"] += 1
            audit["status"] = "exception"
            audit["final_reason"] = "exception"
            audit["error_type"] = type(e).__name__
            audit["error_msg"] = str(e)
            print(f"[CH {ch}] EXCEPTION: {type(e).__name__}: {e}")

        finally:
            audit["t_channel_total_s"] = time.perf_counter() - ch_t0
            if np.isnan(audit["t_km_window_s"]):
                audit["t_km_window_s"] = channel_stage_times.get("step2.choose_adaptive_km_window", np.nan)
            if np.isnan(audit["t_kmeans_extract_s"]):
                audit["t_kmeans_extract_s"] = channel_stage_times.get("step3a.extract_snippets_fast_ram[kmeans]", np.nan)
            if np.isnan(audit["t_step1_s"]):
                audit["t_step1_s"] = channel_stage_times.get("step1.find_valley_and_times", np.nan)

            # _append_audit_record(audit)
            lh_channel_audit.append(audit)
            

            if VERBOSE_TIMING:
                print(f"[CH {ch}] ---- stage timings ----")
                for _label, _dt in channel_stage_times.items():
                    print(f"[CH {ch}]   {_label:<36s} {_dt:7.2f} s")
                print(f"[CH {ch}] TOTAL channel | {audit['t_channel_total_s']:.2f} s")

            # Critical for large all-channel runs: do not accumulate figures
            try:
                plt.close("all")
            except Exception:
                pass


    # ----------------------------
    # End-of-run summaries
    # ----------------------------
    lh_channel_audit_by_ch = {int(rec["ch"]): rec for rec in lh_channel_audit}

    success_records = [rec for rec in lh_channel_audit if rec.get("status") == "success"]
    error_records = [rec for rec in lh_channel_audit if rec.get("status") == "exception"]
    valley_not_accepted_records = [
        rec for rec in lh_channel_audit if rec.get("final_reason") == "valley_not_accepted"
    ]
    valley_overflow_records = [
        rec for rec in lh_channel_audit if rec.get("final_reason") == "valley_count>max"
    ]

    lh_round2_plan = dict(
        exclude_error_channels=np.array([rec["ch"] for rec in error_records], dtype=np.int32),
        exclude_valley_not_accepted_channels=np.array(
            [rec["ch"] for rec in valley_not_accepted_records], dtype=np.int32
        ),
        retry_valley_count_gt_max_channels=np.array(
            [rec["ch"] for rec in valley_overflow_records], dtype=np.int32
        ),
        successful_channels=np.array([rec["ch"] for rec in success_records], dtype=np.int32),
    )

    t_loop1 = time.time()
    print("\n====================")
    print(f"Done. Found {len(lh_units)} successful units.")
    print(f"Elapsed: {t_loop1 - t_loop0:.1f} s")

    print("\nRequested stats:")
    print(f"  successful             : {len(success_records)}")
    print(f"  errored                : {len(error_records)}")
    print(f"  valley_not_accepted    : {len(valley_not_accepted_records)}")
    print(f"  valley_count>max       : {len(valley_overflow_records)}")

    overflow_bins = _build_valley_overflow_summary(
        lh_channel_audit,
        valley_limit=MAX_VALLEY_COUNT,
        bin_step=200,
        hard_cap=1000,
    )
    if len(overflow_bins) > 0:
        print(f"\nValley overflow breakdown (> {MAX_VALLEY_COUNT}):")
        for lo, hi, cnt in overflow_bins:
            if hi is None:
                print(f"  {lo:>4d}+ : {cnt}")
            else:
                print(f"  {lo:>4d}-{hi:<4d} : {cnt}")

    print("\nAll rejection / outcome counts:")
    reason_counts = defaultdict(int)
    for rec in lh_channel_audit:
        key = rec.get("final_reason", "unknown")
        reason_counts[key] += 1
    for k in sorted(reason_counts.keys()):
        print(f"  {k:>24s} : {reason_counts[k]}")

    # ----------------------------
    # End-of-run plots for successful channels
    # ----------------------------
    if PLOT_END_SUMMARY and len(success_records) > 0:
        success_records = sorted(success_records, key=lambda rec: rec["unit_index"])
        unit_order = np.arange(1, len(success_records) + 1, dtype=int)

        t_km_window = np.array([rec["t_km_window_s"] for rec in success_records], dtype=float)
        t_kmeans_extract = np.array([rec["t_kmeans_extract_s"] for rec in success_records], dtype=float)
        km_pre = np.array([rec["km_win_pre"] for rec in success_records], dtype=float)
        km_post = np.array([rec["km_win_post"] for rec in success_records], dtype=float)

        plt.figure(figsize=(12, 4))
        plt.plot(unit_order, t_km_window, marker="o", label="adaptive km_window")
        plt.plot(unit_order, t_kmeans_extract, marker="o", label="extract 5k for k-means")
        plt.xlabel("Successful unit order")
        plt.ylabel("Time (s)")
        plt.title("Adaptive-window timing on successful channels")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(8, max(4, 0.20 * len(success_records) + 2)))
        plt.scatter(km_pre, unit_order, label="pre", s=30)
        plt.scatter(km_post, unit_order, label="post", s=30)
        plt.xlabel("km_window limit (samples)")
        plt.ylabel("Successful unit order")
        plt.title("Adaptive km_window limits")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()



    # save_path = Path("/Volumes/Lab/Users/alexth/axolotl/202510236/data018/lh_run_dump.pkl")
    # save_path = Path("/Volumes/Lab/Users/alexth/axolotl/202510236/lh_run_dump.pkl")
    # save_path = Path("/Volumes/Lab/Users/alexth/axolotl/202310301/data002/lh_run_dump.pkl")
    # save_path = Path("/Volumes/Lab/Users/alexth/axolotl/202512300/data001/lh_run_dump.pkl")
    # save_path = Path("/Volumes/Lab/Users/alexth/axolotl/201808079/data007_lh_run_dump.pkl")
    save_path = os.path.join(raw_dir, exp_name, f'{datafile_name}_lh_run_dump.pkl')


    lh_dump = {
        "lh_units": lh_units,
        "lh_uncertain_ledger": lh_uncertain_ledger,
        "lh_channel_audit": lh_channel_audit,
        "lh_channel_audit_by_ch": lh_channel_audit_by_ch,
        "lh_round2_plan": lh_round2_plan,
        "skip_counts": dict(skip_counts),
        "channels_ran": CHANNELS_TO_RUN,
        "max_valley_count": MAX_VALLEY_COUNT,
    }

    with open(save_path, "wb") as f:
        pickle.dump(lh_dump, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to {save_path}")

if __name__ == "__main__":
    exp_name = '20260415C'
    datafile_name = 'data032'
    main(exp_name, datafile_name)