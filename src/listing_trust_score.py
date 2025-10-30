import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ALPHA, BETA, GAMMA = 0.5, 0.3, 0.2    # weights

def _to_frac(x):
    """Convert '95%' or '0.95' or 95 to fraction in [0,1]."""
    if pd.isna(x): return np.nan
    if isinstance(x, str) and x.strip().endswith('%'):
        try: return float(x.strip('%'))/100.0
        except: return np.nan
    try:
        x = float(x)
        return x/100.0 if x > 1.0 else x
    except:
        return np.nan

def minmax(s: pd.Series):
    s = s.astype(float)
    lo, hi = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        # if degenerate, return zeros
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - lo) / (hi - lo)

def compute_listing_trust(preds: pd.DataFrame,
                          listings: pd.DataFrame,
                          outdir: str = "Figures",
                          sentiment_mode: str = "discrete"  # "discrete" or "positive_prob"
                          ) -> pd.DataFrame:
    """
    Returns a DataFrame with listing_id, D_mean, S_mean, B_j, T_j, n_reviews and saves a histogram figure.
    - preds must have: listing_id, deception_prob_label_1, and either sent_pred or sent_prob_positive
    - listings provides behavioral fields; fallbacks used if some are missing
    """
    os.makedirs(outdir, exist_ok=True)

    # --- Sentiment score per review ---
    if sentiment_mode == "discrete" and "sent_pred" in preds.columns:
        # map 0/1/2 -> 0.0 / 0.5 / 1.0
        sent_map = {0: 0.0, 1: 0.5, 2: 1.0}
        preds = preds.copy()
        preds["sent_score"] = preds["sent_pred"].map(sent_map).astype(float)
    elif "sent_prob_positive" in preds.columns:
        preds = preds.copy()
        preds["sent_score"] = preds["sent_prob_positive"].astype(float)
    else:
        raise ValueError("Need either sent_pred (for discrete) or sent_prob_positive in preds.")

    # --- Aggregate to listing level ---
    agg = (preds.groupby("listing_id", as_index=False)
                .agg(D_mean=("deception_prob_label_1", "mean"),
                     S_mean=("sent_score", "mean"),
                     n_reviews=("listing_id", "size")))

    # --- Behavioral term B_j from listings (robust to what's available) ---
    # Candidate features: number_of_reviews, estimated_occupancy_l365d, host_response_rate
    behav = listings[["id", "number_of_reviews", "estimated_occupancy_l365d", "host_response_rate"]].copy() \
                if set(["id","number_of_reviews","estimated_occupancy_l365d","host_response_rate"]).issubset(listings.columns) \
                else listings[["id", "number_of_reviews"]].copy()

    behav = behav.rename(columns={"id": "listing_id"})
    if "host_response_rate" in behav.columns:
        behav["host_response_rate_frac"] = behav["host_response_rate"].apply(_to_frac)
    if "number_of_reviews" in behav.columns:
        behav["number_of_reviews"] = pd.to_numeric(behav["number_of_reviews"], errors="coerce")
    if "estimated_occupancy_l365d" in behav.columns:
        behav["estimated_occupancy_l365d"] = pd.to_numeric(behav["estimated_occupancy_l365d"], errors="coerce")

    # Normalize available components and average (ignore missing)
    components = []
    if "number_of_reviews" in behav.columns: components.append(minmax(behav["number_of_reviews"]))
    if "estimated_occupancy_l365d" in behav.columns: components.append(minmax(behav["estimated_occupancy_l365d"]))
    if "host_response_rate_frac" in behav.columns: components.append(minmax(behav["host_response_rate_frac"]))

    if components:
        B_norm = pd.concat(components, axis=1).mean(axis=1, skipna=True)
        behav["B_j"] = B_norm.fillna(0.0)
    else:
        # Fallback: use normalized review count from preds aggregation
        tmp = agg[["listing_id", "n_reviews"]].copy()
        tmp["B_j"] = minmax(tmp["n_reviews"])
        behav = tmp

    # Merge behavioral term onto agg; if B_j missing, default to normalized n_reviews
    listing_df = agg.merge(behav[["listing_id", "B_j"]], on="listing_id", how="left")
    if listing_df["B_j"].isna().any():
        listing_df["B_j"] = listing_df["B_j"].fillna(minmax(listing_df["n_reviews"]))

    # --- Compute T_j ---
    listing_df["T_j"] = (ALPHA * (1.0 - listing_df["D_mean"])
                         + BETA * listing_df["S_mean"]
                         + GAMMA * listing_df["B_j"])

    # Save CSV for appendix/repro
    listing_df.sort_values("T_j", ascending=False).to_csv(
        os.path.join(outdir, "listing_trust_scores.csv"), index=False
    )

    # --- Plot distribution ---
    plt.figure(figsize=(8,5))
    plt.hist(listing_df["T_j"], bins=50, edgecolor="black")
    plt.title("Distribution of Listing-Level Trust Scores ($T_j$)")
    plt.xlabel("Trust score ($T_j$)")
    plt.ylabel("Number of listings")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "listing_trust_distribution.png"), dpi=300)
    plt.show()

    return listing_df