# sentimental-analysis-chat-bot
# 🤖 Sentiment Analysis Chatbot

A machine learning–powered chatbot that analyzes Amazon product reviews and classifies them as **Positive**, **Neutral**, or **Negative** — then replies with a context-aware, empathetic response through a simple desktop GUI.

Built as part of the *Fundamentals of AI* coursework at Amrita Vishwa Vidyapeetham.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Overview

Businesses drown in customer reviews. This project turns raw text feedback into instant, actionable sentiment — and responds to the user in real time, like a support agent that never sleeps.

Two models were built and compared:

| Model | Technique | Accuracy |
|---|---|---|
| **Random Forest** ✅ *(used in final app)* | TF-IDF + Random Forest, with a custom keyword bag-of-words layer for fast neutral/negative detection | **~88.5%** |
| LSTM | Bidirectional LSTM + Conv1D over tokenized/padded sequences | ~64% |

Random Forest also outperformed KNN, SVM, Decision Tree, and Multinomial Naive Bayes in head-to-head accuracy/precision/recall comparisons — making it the model powering the final chatbot.

## ✨ Features

- 🧠 **TF-IDF + Random Forest** sentiment classifier (positive / neutral / negative)
- ⚡ **Custom keyword shortcuts** for common neutral/negative phrases → faster, more decisive responses
- 💬 **Randomized, human-like replies** so the chatbot doesn't feel robotic
- 🖥️ **Tkinter GUI** — type a review, hit submit, get instant feedback
- 📊 **Live accuracy visualization** plotted after training
- 🧪 An experimental **LSTM** pipeline included for comparison

## 🗂️ Repository Structure
## 📊 Dataset

- **Source:** [Amazon Review Data (Electronics)](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) — Julian McAuley, UCSD
- **Size:** 10,000 entries, 2 columns
- **Columns:**
  - `overall` — integer rating (1–5), used as a sentiment proxy
  - `reviewText` — the review text itself
- **Missing values:** 6 out of 10,000 (negligible)

## 🛠️ Tech Stack

- **Language:** Python 3
- **ML/NLP:** scikit-learn (TF-IDF, RandomForestClassifier), NLTK (tokenization, stopwords, lemmatization), TensorFlow/Keras (LSTM)
- **GUI:** Tkinter
- **Visualization:** Matplotlib

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Deepthi-060106/sentimental-analysis-chat-bot.git
cd sentimental-analysis-chat-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data (first run only)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Run it
Open `FAIFinalCode.ipynb` in Jupyter and run the **Random Forest** cell to launch the Tkinter chatbot GUI (this is the primary, best-performing model). The LSTM cell above it is included for comparison/experimentation.

> **Note:** the notebook currently loads the dataset from a hardcoded local path (`review_data(Sheet1).csv`). Update the path in the first code cell of each section to point to the CSV in this repo before running.

## 📈 Results

- **Random Forest (final model):** ~88.5% accuracy on held-out test data, with balanced precision and recall across positive/neutral/negative classes.
- **LSTM (experimental):** ~64% accuracy — struggled to generalize sentiment patterns as well as the TF-IDF + tree-ensemble approach on this dataset.
- Random Forest also beat KNN, SVM, Decision Tree, and Naive Bayes baselines in side-by-side comparison.

Full methodology, confusion matrices, and discussion are in [`FAI_FinalReport.pdf`](./FAI_FinalReport.pdf).

## 👥 Team A-14

| Name | Roll No. |
|---|---|
| A. Deepthi | CB.SC.U4AIE23007 |
| A. Sai Sanjana | CB.SC.U4AIE23013 |
| K. Prerana | CB.SC.U4AIE23038 |
| P. Srikari Shasi | CB.SC.U4AIE23053 |
| Rithish A B | CB.SC.U4AIE23057 |

*Fundamentals of AI (22AIE201) — Amrita School of Artificial Intelligence, Coimbatore*

## 🔮 Future Improvements

- Better handling of neutral sentiment misclassification
- Replace hardcoded file paths with relative/configurable paths
- Package the GUI as a standalone executable
- Swap Tkinter for a web-based interface (Streamlit/Flask)

## 📄 License

This project is for academic purposes. Feel free to fork and build on it — attribution appreciated.
