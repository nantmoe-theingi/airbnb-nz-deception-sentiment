# distribution_deception_sentiment_predictions.py
# ------------------------------------------------------------
# Figures for: Distribution of Deception and Sentiment Predictions
# Call these from a notebook. No seaborn; every plot saves + shows.
# ------------------------------------------------------------

import os
from typing import Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ---------- Config ----------
DEFAULT_OUTDIR = "figures"
REQUIRED_COLS: Sequence[str] = (
    "deception_prob_label_1",
    "sent_prob_negative",
    "sent_prob_neutral",
    "sent_prob_positive",
    "date",
)


# ---------- I/O & Validation ----------
def ensure_outdir(path: str = DEFAULT_OUTDIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def validate_columns(df: pd.DataFrame, required: Sequence[str] = REQUIRED_COLS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    z = df.copy()
    for c in cols:
        if c in z.columns:
            z[c] = pd.to_numeric(z[c], errors="coerce")
    return z


def coerce_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    z = df.copy()
    z[date_col] = pd.to_datetime(z[date_col], errors="coerce")
    return z.dropna(subset=[date_col])


# ---------- Plots ----------
def plot_deception_histogram(
    df: pd.DataFrame,
    prob_col: str = "deception_prob_label_1",
    outdir: str = DEFAULT_OUTDIR,
    bins: int = 50,
    show_threshold: bool = True,
    threshold: Optional[float] = 0.81,
    title: str = "Distribution of Deception Probability",
) -> str:
    """
    Histogram of P(deceptive). Saves PNG and shows the figure.
    """
    ensure_outdir(outdir)
    s = pd.to_numeric(df[prob_col], errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=bins, range=(0, 1))
    if show_threshold and threshold is not None:
        plt.axvline(threshold, linestyle="--")
        plt.text(
            threshold,
            plt.gca().get_ylim()[1] * 0.95,
            f"τ = {threshold:.2f}",
            ha="right",
            va="top",
        )
    plt.title(title)
    plt.xlabel("P(deceptive)")
    plt.ylabel("Count")
    plt.tight_layout()

    out_path = os.path.join(outdir, "deception_probability_hist.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()
    return out_path


def monthly_sentiment_means(
    df: pd.DataFrame,
    date_col: str = "date",
    neg_col: str = "sent_prob_negative",
    neu_col: str = "sent_prob_neutral",
    pos_col: str = "sent_prob_positive",
) -> pd.DataFrame:
    """
    Monthly mean sentiment probabilities; rows are first day of each month (MS).
    Re-normalizes each row to sum to 1.
    """
    z = coerce_datetime(df, date_col=date_col)
    z = coerce_numeric(z, [neg_col, neu_col, pos_col])
    monthly = z.set_index(date_col)[[neg_col, neu_col, pos_col]].resample("MS").mean()
    monthly = monthly.clip(lower=0)
    row_sums = monthly.sum(axis=1)
    monthly = monthly.div(row_sums.where(row_sums > 0, np.nan), axis=0)
    monthly.index.name = "month"
    return monthly


def plot_monthly_sentiment_stack(
    df: pd.DataFrame,
    date_col: str = "date",
    neg_col: str = "sent_prob_negative",
    neu_col: str = "sent_prob_neutral",
    pos_col: str = "sent_prob_positive",
    outdir: str = DEFAULT_OUTDIR,
    title: str = "Sentiment Probability Composition (Monthly Means)",
) -> Tuple[str, str]:
    """
    Stacked area (0–100%) of monthly mean sentiment probabilities.
    Saves CSV + PNG and shows the figure.
    """
    ensure_outdir(outdir)
    monthly = monthly_sentiment_means(df, date_col, neg_col, neu_col, pos_col)

    # Save CSV
    out_csv = os.path.join(outdir, "monthly_sentiment_means.csv")
    monthly.to_csv(out_csv)

    # Plot
    x = monthly.index
    neg = monthly[neg_col].values
    neu = monthly[neu_col].values
    pos = monthly[pos_col].values

    plt.figure(figsize=(9, 5.2))
    plt.stackplot(x, neg, neu, pos, labels=["Negative", "Neutral", "Positive"])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Share")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()

    out_png = os.path.join(outdir, "sentiment_composition_stacked_area.png")
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close()
    return out_csv, out_png


def monthly_dual_trend(
    df: pd.DataFrame,
    date_col: str = "date",
    deception_col: str = "deception_prob_label_1",
    pos_col: str = "sent_prob_positive",
) -> pd.DataFrame:
    """
    Monthly means for deception probability and positive sentiment probability.
    """
    z = coerce_datetime(df, date_col=date_col)
    z = coerce_numeric(z, [deception_col, pos_col])
    monthly = (
        z.set_index(date_col)[[deception_col, pos_col]]
        .resample("MS")
        .mean()
        .rename(
            columns={
                deception_col: "mean_deception_prob",
                pos_col: "mean_positive_prob",
            }
        )
    )
    monthly.index.name = "month"
    return monthly


def plot_monthly_dual_trend(
    df: pd.DataFrame,
    date_col: str = "date",
    deception_col: str = "deception_prob_label_1",
    pos_col: str = "sent_prob_positive",
    outdir: str = DEFAULT_OUTDIR,
    title: str = "Monthly Trends: Mean Deception vs Mean Positive Sentiment",
) -> Tuple[str, str]:
    """
    Dual-line monthly trends (mean deception vs mean positive sentiment).
    Saves CSV + PNG and shows the figure.
    """
    ensure_outdir(outdir)
    monthly = monthly_dual_trend(df, date_col, deception_col, pos_col)

    # Save CSV
    out_csv = os.path.join(outdir, "monthly_dual_trends.csv")
    monthly.to_csv(out_csv)

    # Plot
    plt.figure(figsize=(9, 5.2))
    plt.plot(monthly.index, monthly["mean_deception_prob"], label="Mean P(deceptive)")
    plt.plot(monthly.index, monthly["mean_positive_prob"], label="Mean P(positive)")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Probability")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()

    out_png = os.path.join(outdir, "monthly_dual_trends.png")
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close()
    return out_csv, out_png


def run_all(
    preds: pd.DataFrame,
    outdir: str = DEFAULT_OUTDIR,
    threshold: Optional[float] = 0.81,
) -> dict:
    """
    Quick helper to generate all outputs in one call.
    """
    validate_columns(preds)
    paths = {}
    paths["deception_hist_png"] = plot_deception_histogram(
        preds, outdir=outdir, threshold=threshold
    )
    paths["sent_stack_csv"], paths["sent_stack_png"] = plot_monthly_sentiment_stack(
        preds, outdir=outdir
    )
    paths["dual_csv"], paths["dual_png"] = plot_monthly_dual_trend(preds, outdir=outdir)
    return paths