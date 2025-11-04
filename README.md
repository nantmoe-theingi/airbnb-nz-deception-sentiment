# Airbnb NZ — Deception & Sentiment Analysis using SVM and DistilBERT

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow)


## Overview
This project investigates how artificial intelligence can enhance trust and credibility evaluation in peer-to-peer (P2P) digital platforms.  
It applies **deception detection** and **sentiment analysis** models to large-scale Airbnb New Zealand review data, forming part of a broader trust-framework study.


## Objectives
- Develop and compare **SVM** and **DistilBERT** models for text classification.  
- Detect **deceptive vs truthful** reviews using the *Deceptive Opinion Spam Corpus*.  
- Identify **positive / neutral / negative** sentiment using the *TripAdvisor Hotel Reviews* dataset.  
- Apply both models to **Airbnb New Zealand** reviews to infer credibility patterns at scale.


## Pipeline
1. **SVM (baseline)** — deception detection on *Deceptive Opinion Spam Corpus* (Ott et al.)  
2. **DistilBERT (transformer)** — deception detection; inference on *Airbnb NZ*  
3. **SVM (baseline)** — sentiment classification on *TripAdvisor* reviews  
4. **DistilBERT (transformer)** — sentiment classification; inference on *Airbnb NZ*  
5. **Integrated Analysis** — combine deception + sentiment scores for credibility and trust insights

## Implementation
- **Language:** Python 3.10  
- **Frameworks:** scikit-learn, PyTorch, Hugging Face Transformers  
- **Hardware:** NVIDIA GPU 
- **Output:** Chunked inference → Parquet results  


## Key Results
- Transformer-based **DistilBERT** outperformed traditional **SVM** in both deception and sentiment tasks.  
- Large-scale inference on **1.5 M+ Airbnb NZ reviews** achieved ≈ 493 rows/s on GPU (FP16 precision).  
- Outputs support higher-level trust and reputation modelling in P2P ecosystems.

## Datasets
- **Deceptive Opinion Spam Corpus** (Ott et al.) — deception detection  
- **TripAdvisor Hotel Reviews** — sentiment classification  
- **Airbnb New Zealand Reviews** — application dataset from [Inside Airbnb](https://insideairbnb.com)

> Raw datasets are publicly available from their original sources.  
> This repository includes **code and documentation only**.

## Data & Models (keep out of Git)
Store large files in Google Drive.
