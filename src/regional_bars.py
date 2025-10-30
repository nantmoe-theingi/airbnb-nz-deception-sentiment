# regional_bars.py
# ------------------------------------------------------------
# Regional top-N bar charts for Airbnb inference outputs
# ------------------------------------------------------------

import os
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def top_regions_bar(
    preds: pd.DataFrame,
    region_col: str = "region_name",
    dece_col: str = "deception_prob_label_1",
    value: str = "count",            # "count" or "mean_deception"
    top_n: int = 10,
    min_reviews: int = 50,           # only used for mean_deception
    outdir: str = "figures",
    fig_name: Optional[str] = None,
    as_percent_for_mean: bool = True # format x-axis as % for mean_deception
) -> Tuple[str, str]:
    """
    Plot top-N regions by review count or by mean deception probability.

    Returns (fig_path).
    """
    _ensure_outdir(outdir)

    if region_col not in preds.columns:
        raise ValueError(f"Column '{region_col}' not found in DataFrame.")

    df = preds.dropna(subset=[region_col]).copy()

    if value == "count":
        agg_series = df.groupby(region_col).size().sort_values(ascending=False).head(top_n)
        ylabel = "Number of reviews"
        title = f"Top {top_n} Regions by Review Count"
        default_fig = f"top{top_n}_regions_by_count.png"

    elif value == "mean_deception":
        if dece_col not in df.columns:
            raise ValueError(
                f"value='mean_deception' requires column '{dece_col}' in DataFrame."
            )
        # filter tiny-sample regions
        counts = df.groupby(region_col).size()
        keep = counts[counts >= min_reviews].index
        z = df[df[region_col].isin(keep)].copy()

        z[dece_col] = pd.to_numeric(z[dece_col], errors="coerce")

        agg_series = (
            z.groupby(region_col)[dece_col]
             .mean()
             .sort_values(ascending=False)
             .head(top_n)
        )
        ylabel = "Mean deception probability"
        title = f"Top {top_n} Regions by Mean Deception Probability"
        default_fig = f"top{top_n}_regions_by_mean_deception.png"
    else:
        raise ValueError("value must be 'count' or 'mean_deception'")



    # ----- Plot (use Series so index=region labels appear on y-axis) -----
    plt.figure(figsize=(9, 5))
    s = agg_series.sort_values(ascending=True)  # smallest at bottom, largest at top
    ax = s.plot(kind="barh", color="#3B82F6", edgecolor="black", width=0.8)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Region")
    plt.tight_layout()

    if value == "mean_deception" and as_percent_for_mean:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0, 1)

    out_png = os.path.join(outdir, fig_name or default_fig)
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved figure to: {out_png}")

    return out_png