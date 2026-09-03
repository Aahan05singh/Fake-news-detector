# Fake-news-detector
Detects fake vs. real news articles using NLP + Logistic Regression, with a bonus digital-signature verification demo.

A machine learning project that classifies news articles as **Real** or **Fake** using natural language processing (NLP) and a Logistic Regression classifier. The project also includes an experimental module demonstrating how digital signatures (RSA + SHA-256) could be used to cryptographically verify the authenticity of a news article.

## Overview

This project uses the classic **True/Fake news dataset** (article title, text, subject, and date) and applies a standard NLP pipeline:

1. **Text cleaning** — lowercasing and punctuation removal
2. **Tokenization** — splitting text into word tokens
3. **Stopword removal** — using NLTK's English stopword list
4. **Lemmatization** — reducing words to their base form with WordNet
5. **Feature extraction** — TF-IDF vectorization (unigrams, top 5,000 features)
6. **Classification** — Logistic Regression trained on TF-IDF features
7. **Evaluation** — accuracy, confusion matrix, and classification report

As a bonus, the notebook also explores a **digital signature demo**: hashing an article with SHA-256, signing the hash with an RSA private key, and verifying it with the corresponding public key — illustrating one way news authenticity could be cryptographically attested to, independent of the ML classifier.

## Results

The Logistic Regression model achieves **~99% accuracy** on the held-out test set (20% split), with balanced precision/recall across both classes.

| Metric | Score |
|---|---|
| Accuracy | ~0.99 |
| Precision (Fake / Real) | ~0.99 / 0.99 |
| Recall (Fake / Real) | ~0.99 / 0.99 |

> Exact numbers will vary slightly depending on the `random_state` and dataset version used.

## Project Structure

```
fake-news-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── notebooks/
│   └── fake_news_detection.ipynb   # Original exploratory notebook (from Colab)
├── src/
│   ├── train.py                    # Script version: trains and saves the model
│   └── predict.py                  # Script version: loads the model and classifies new text
└── data/
    └── README.md                   # Where to download the dataset
```

## Dataset

This project uses the **ISOT Fake News Dataset**, made up of two CSV files:

- `True.csv` — real news articles
- `Fake.csv` — fake news articles

The dataset is **not included** in this repository (see `data/README.md` for download instructions and licensing notes).

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download NLTK data (first run only)

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4. Get the dataset

Download `True.csv` and `Fake.csv` (see `data/README.md`) and place them in the `data/` folder.

### 5. Train the model

```bash
python src/train.py
```

This will train the model and save `model.joblib` and `tfidf_vectorizer.joblib` to the project root (or a `models/` folder — see script).

### 6. Run predictions

```bash
python src/predict.py --text "The Senate passed a new infrastructure bill on Tuesday."
```

Or explore the full pipeline interactively in `notebooks/fake_news_detection.ipynb`.

## Tech Stack

- **Python 3**
- **pandas / numpy** — data handling
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, evaluation metrics
- **NLTK** — stopword removal and lemmatization
- **matplotlib / seaborn** — visualization
- **cryptography** — RSA digital signature demo

## Limitations & Future Work

- The classifier is trained on a specific dataset (largely US political news from 2016–2017) and may not generalize well to other domains, time periods, or writing styles.
- TF-IDF + Logistic Regression is a strong baseline but doesn't capture context the way transformer-based models (e.g., BERT) can — a natural next step.
- The digital signature module is a standalone proof-of-concept and is not yet integrated into the prediction pipeline.
- No hyperparameter tuning or cross-validation was performed; this is a good next improvement.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
