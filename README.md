# Airbnb NZ — Deception & Sentiment with SVM and DistilBERT

**Goal.** Build and compare SVM and DistilBERT for:
- **Deception**: fake vs. true (train on Deceptive Opinion dataset; predict on Airbnb NZ)
- **Sentiment**: positive / neutral / negative (train on TripAdvisor; predict on Airbnb NZ)

## Pipeline
1) **SVM (baseline) — deception** on Deceptive Opinion (Ott et al.)  
2) **DistilBERT — deception** on Deceptive Opinion; predict on Airbnb NZ  
3) **SVM (baseline) — sentiment** on TripAdvisor (review + rating → pos/neu/neg)  
4) **DistilBERT — sentiment** on TripAdvisor; predict on Airbnb NZ  
5) **Analysis on Airbnb NZ** (language filter=English, chunked inference, Parquet outputs)

## Key Analysis Questions
- **Sentiment Extremity:** Are predicted fake reviews more often extreme (1★/5★) than true reviews?
- **Punctuation/Capitalization:** Do fake reviews overuse “!” or ALL CAPS?
- **Review Length:** Is length different between predicted fake vs. true?
- **Temporal Patterns:** Spikes or regular intervals in predicted fake posting times?
- **Truthful vs. Deceptive Sentiment:** Are truthful reviews more neutral/balanced?
- **Deceptive Sentiment Profile:** Mostly positive, negative, or U-shaped (1★ & 5★ heavy)?
- **Which Sentiment Contains Most Deception?** (pos/neu/neg)
- **Language Cues:** Differences between “Deceptive-Negative” vs “Truthful-Negative.”

## Repo Layout
- `notebooks/01_svm_deception.ipynb` — TF-IDF + LinearSVM baseline  
- `notebooks/02_distilbert_deception.ipynb` — fine-tune & eval  
- `notebooks/03_predict_airbnb_deception.ipynb` — chunked DistilBERT inference on Airbnb NZ  
- `notebooks/04_svm_tripadvisor_sentiment.ipynb` — TF-IDF + SVM 3-class  
- `notebooks/05_distilbert_tripadvisor_sentiment.ipynb` — 3-class fine-tune & eval  
- `notebooks/06_predict_airbnb_sentiment.ipynb` — chunked 3-class inference  
- `notebooks/07_airbnb_analysis.ipynb` — all analysis questions

## Data & Models (keep out of Git)
Store large files in Google Drive:
