import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.sparse import csr_matrix
from scipy.sparse import diags

def _sent_score_from_preds(df, mode="discrete"):
    """Return a 1D numpy array of per-row sentiment scores in [0,1]."""
    if mode == "discrete" and "sent_pred" in df.columns:
        # 0/1/2 -> 0.0 / 0.5 / 1.0
        m = {0: 0.0, 1: 0.5, 2: 1.0}
        return df["sent_pred"].map(m).astype(float).to_numpy()
    elif "sent_prob_positive" in df.columns:
        return df["sent_prob_positive"].astype(float).to_numpy()
    else:
        raise ValueError("Need sent_pred (discrete) or sent_prob_positive in preds.")

def _safe_minmax(s: pd.Series):
    s = s.astype(float)
    lo, hi = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - lo) / (hi - lo)

def propagate_platform_trust(
    preds: pd.DataFrame,
    reviews: pd.DataFrame,
    listings: pd.DataFrame,
    listing_trust: pd.DataFrame | None = None,  # optional, with columns ["listing_id","T_j"]
    sentiment_mode: str = "discrete",           # "discrete" or "positive_prob"
    lam: float = 0.6,                           # lambda in the paper text
    eps: float = 1e-4,
    max_iter: int = 50,
    outdir: str = "figures",
    make_region_plot: bool = True
):
    """
    Returns (final_listing_trust_df, final_reviewer_trust_df).
    Also saves two CSVs and (optionally) a regional bar plot.
    """

    os.makedirs(outdir, exist_ok=True)

    # --- 1) Join preds with reviews to get reviewer_id (preds must have review_id) ---
    if "review_id" not in preds.columns:
        raise ValueError("preds must include 'review_id' to join reviewer_id from reviews.")
    if "id" not in reviews.columns:
        raise ValueError("reviews must include primary key 'id' for review_id.")

    edge_df = preds.merge(
        reviews[["id", "reviewer_id"]],
        left_on="review_id", right_on="id", how="inner",
        suffixes=("", "_r")
    ).drop(columns=["id"])  # 'id' from reviews

    # Require listing_id and deception prob
    if "listing_id" not in edge_df.columns:
        raise ValueError("preds must include 'listing_id'.")
    if "deception_prob_label_1" not in edge_df.columns:
        raise ValueError("preds must include 'deception_prob_label_1' (P(deceptive)).")

    # --- 2) Build reviewer baseline r0 = 1 - mean P(deceptive) ---
    rev_grp = edge_df.groupby("reviewer_id")["deception_prob_label_1"].mean()
    r0 = (1.0 - rev_grp).clip(0, 1)  # reviewer baseline credibility
    reviewers = r0.index.to_numpy()
    R = len(reviewers)
    rev_index = {rid: i for i, rid in enumerate(reviewers)}

    # --- 3) Build listing baseline t0 ---
    if listing_trust is not None and {"listing_id","T_j"}.issubset(listing_trust.columns):
        lt0 = listing_trust.set_index("listing_id")["T_j"]
    else:
        # Fallback: α*(1 - D_mean) + β*S_mean + γ*B (with B=f(normalized review count))
        alpha, beta, gamma = 0.5, 0.3, 0.2
        # listing means from preds
        tmp = edge_df.copy()
        tmp["sent_score"] = _sent_score_from_preds(tmp, mode=sentiment_mode)
        byL = tmp.groupby("listing_id").agg(
            D_mean=("deception_prob_label_1", "mean"),
            S_mean=("sent_score", "mean"),
            n_reviews=("listing_id", "size"),
        )
        byL["B"] = _safe_minmax(byL["n_reviews"])
        lt0 = (alpha * (1.0 - byL["D_mean"]) + beta * byL["S_mean"] + gamma * byL["B"]).clip(0, 1)

    listings_ids = lt0.index.to_numpy()
    L = len(listings_ids)
    list_index = {lid: j for j, lid in enumerate(listings_ids)}

    # --- 4) Build sparse edge matrix W (reviewer -> listing) with strength s_ij ---
    # Keep only edges whose (reviewer, listing) both exist in the baselines
    edge_df = edge_df[edge_df["reviewer_id"].isin(reviewers) &
                      edge_df["listing_id"].isin(listings_ids)].copy()

    s_sent = _sent_score_from_preds(edge_df, mode=sentiment_mode)
    s_edge = (1.0 - edge_df["deception_prob_label_1"].astype(float).to_numpy()) * s_sent
    # Map ids to integer indices
    row_i = edge_df["reviewer_id"].map(rev_index).to_numpy()
    col_j = edge_df["listing_id"].map(list_index).to_numpy()

    W = csr_matrix((s_edge, (row_i, col_j)), shape=(R, L))

    # --- 5) Build A (row-stochastic) and B (column-stochastic) ---
    # A: each reviewer row sums to 1 (influence from listings to reviewer)
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    A = diags(1.0 / row_sums) @ W  # R x L

    # B: each listing column sums to 1 (influence from reviewers to listing)
    col_sums = np.array(W.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    B = W @ diags(1.0 / col_sums)  # R x L (column-stochastic)

    # --- 6) Iterative updates ---
    r = r0.reindex(reviewers).to_numpy()
    t = lt0.reindex(listings_ids).to_numpy()
    r0_vec = r.copy()
    t0_vec = t.copy()

    for it in range(max_iter):
        r_next = (1.0 - lam) * r0_vec + lam * (A @ t)
        t_next = (1.0 - lam) * t0_vec + lam * (B.T @ r)

        delta = max(np.max(np.abs(r_next - r)), np.max(np.abs(t_next - t)))
        r, t = r_next, t_next
        if delta < eps:
            # print(f"Converged in {it+1} iterations (Δ={delta:.2e}).")
            break
    # else:
    #     print(f"Stopped at max_iter={max_iter} (Δ={delta:.2e}).")

    # --- 7) Package outputs ---
    out_reviewers = pd.DataFrame({"reviewer_id": reviewers, "trust_reviewer": r})
    out_listings = pd.DataFrame({"listing_id": listings_ids, "trust_listing": t})

    # Save CSVs
    out_reviewers.sort_values("trust_reviewer", ascending=False).to_csv(
        os.path.join(outdir, "platform_reviewer_trust.csv"), index=False
    )
    out_listings.sort_values("trust_listing", ascending=False).to_csv(
        os.path.join(outdir, "platform_listing_trust.csv"), index=False
    )

    # --- 8) Optional: regional aggregation plot (bar) ---
    if make_region_plot and "region" in listings.columns:
        region_map = listings[["id", "region"]].rename(columns={"id":"listing_id"})
        reg = out_listings.merge(region_map, on="listing_id", how="left")
        reg_mean = (reg.groupby("region", dropna=False)["trust_listing"]
                        .mean().sort_values(ascending=False).head(15))
        plt.figure(figsize=(9,5))
        reg_mean.iloc[::-1].plot(kind="barh")  # top->bottom
        plt.title("Platform-Level Trust (mean $T_j$) by Region — Top 15")
        plt.xlabel("Mean Listing Trust after Propagation")
        plt.ylabel("Region")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "regional_trust_bar.png"), dpi=300)
        plt.show()

    return out_listings, out_reviewers
