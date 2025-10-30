import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_reviewer_credibility(preds: pd.DataFrame, reviews: pd.DataFrame, outdir: str = "Figures") -> pd.DataFrame:
    """
    Merge reviewer_id into preds, compute reviewer credibility, and plot figures.
    Returns a DataFrame of reviewer-level credibility statistics.
    """

    os.makedirs(outdir, exist_ok=True)

    # --- 1) Map reviewer_id from reviews to preds ---
    key_map = (
        reviews[["id", "reviewer_id"]]
        .dropna(subset=["id", "reviewer_id"])
        .drop_duplicates(subset=["id"])
        .rename(columns={"id": "review_id"})
    )
    
    # Ensure both merge keys are the same dtype 
    key_map["review_id"] = key_map["review_id"].astype(str)
    preds["review_id"] = preds["review_id"].astype(str)
    
    preds_with_rid = preds.merge(key_map, on="review_id", how="left")
    
    

    missing = preds_with_rid["reviewer_id"].isna().sum()
    total = len(preds_with_rid)
    print(f"Reviewer IDs attached: {total - missing:,}/{total:,}  (missing: {missing:,})")

    preds_with_rid = preds_with_rid.dropna(subset=["reviewer_id"])
    preds_with_rid["deception_prob_label_1"] = pd.to_numeric(
        preds_with_rid["deception_prob_label_1"], errors="coerce"
    )

    # --- 2) Compute reviewer-level credibility ---
    df_reviewers = (
        preds_with_rid.groupby("reviewer_id", as_index=False)["deception_prob_label_1"]
        .mean()
        .rename(columns={"deception_prob_label_1": "mean_deception_prob"})
    )
    df_reviewers["credibility"] = 1 - df_reviewers["mean_deception_prob"]

    # Add review counts
    counts = preds_with_rid["reviewer_id"].value_counts().rename_axis("reviewer_id").reset_index(name="review_count")
    df_reviewers = df_reviewers.merge(counts, on="reviewer_id", how="left")

    # Save summary table
    out_csv = os.path.join(outdir, "reviewer_credibility_table.csv")
    df_reviewers.to_csv(out_csv, index=False)
    print(f"Saved reviewer credibility table → {out_csv}")

    # --- 3) Plot histogram ---
    plt.figure(figsize=(8,5))
    plt.hist(df_reviewers["credibility"], bins=50, color="skyblue", edgecolor="black")
    plt.title("Distribution of Reviewer Credibility Scores")
    plt.xlabel("Reviewer Credibility (1 − mean deception probability)")
    plt.ylabel("Number of Reviewers")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reviewer_credibility_distribution.png"), dpi=300)
    plt.show()

    # --- 4) Plot activity vs credibility ---
    plt.figure(figsize=(8,5))
    plt.scatter(df_reviewers["review_count"], df_reviewers["credibility"], alpha=0.35)
    plt.xscale("log")
    plt.title("Reviewer Activity vs. Credibility")
    plt.xlabel("Number of Reviews (log scale)")
    plt.ylabel("Reviewer Credibility")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reviewer_activity_vs_credibility.png"), dpi=300)
    plt.show()

    return df_reviewers