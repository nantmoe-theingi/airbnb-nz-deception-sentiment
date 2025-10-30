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

## Data & Models (keep out of Git)
Store large files in Google Drive:
