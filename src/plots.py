"""Diagnostic plots written to results/plots/."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pr_curves(curves: dict, outdir, fname="pr_curves.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=name)
    ax.set_xlabel("Recall (fraud)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - held-out test")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, fname))


def roc_curves(curves: dict, outdir, fname="roc_curves.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC - held-out test")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, fname))


def calibration_plot(curves: dict, outdir, fname="calibration.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud fraction")
    ax.set_title("Calibration - held-out test")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, fname))


def routing_plot(p0, y, band, outdir, fname="routing.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, min(1.0, np.quantile(p0, 0.999)), 40)
    ax.hist(p0[y == 0], bins=bins, alpha=0.6, label="legit", density=True)
    ax.hist(p0[y == 1], bins=bins, alpha=0.6, label="fraud", density=True)
    ax.axvspan(band["tau_low"], band["tau_high"], color="orange", alpha=0.2,
               label="ambiguous band")
    ax.set_xlabel("calibrated LightGBM score p0")
    ax.set_ylabel("density")
    ax.set_title("Ambiguity router")
    ax.legend(fontsize=8)
    _save(fig, os.path.join(outdir, fname))


def bar_compare(names, values, title, ylabel, outdir, fname, ref=None):
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    colors = ["#3b7dd8" if v >= (ref or -1e9) else "#d8663b" for v in values]
    ax.bar(names, values, color=colors)
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=1, label=f"ref={ref:.4f}")
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    _save(fig, os.path.join(outdir, fname))


def shots_plot(shots, auprc, outdir, fname="shots_robustness.png", ideal=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(shots, auprc, "o-")
    if ideal is not None:
        ax.axhline(ideal, color="k", ls="--", label="ideal statevector")
        ax.legend(fontsize=8)
    ax.set_xlabel("shots")
    ax.set_ylabel("AUPRC (ambiguous band)")
    ax.set_title("QRC shot-noise robustness")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, fname))
