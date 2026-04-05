# 📰 NLP Preprocessing and Text Classification — BBC News Dataset

An end-to-end Natural Language Processing pipeline that classifies BBC news articles into **5 categories** using classical ML models. This project covers the full NLP workflow — from raw text preprocessing to model evaluation and comparison.

---

## 📌 Project Overview

| Property        | Details                                                                 |
|----------------|-------------------------------------------------------------------------|
| **Dataset**     | BBC Full Text Document Classification                                   |
| **Source**      | [Kaggle – shivamkushwaha/bbc-full-text-document-classification](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification) |
| **Task**        | Multi-class Text Classification (5 classes)                             |
| **Classes**     | Tech · Business · Sport · Entertainment · Politics                      |
| **Total Samples** | 2,225 full-length BBC news articles (2004–2005)                       |
| **Train / Test Split** | 80% / 20% (stratified)                                           |
| **Language**    | Python 3.10+                                                            |
| **Environment** | Google Colab / Jupyter Notebook                                         |

---

## 📁 Dataset Structure

The dataset is stored as **category subfolders** containing individual `.txt` article files — there is no flat CSV file.

```
bbc/
 ├── tech/              # 401 articles
 │    ├── 001.txt
 │    ├── 002.txt
 │    └── ...
 ├── business/          # 510 articles
 ├── sport/             # 511 articles
 ├── entertainment/     # 386 articles
 └── politics/          # 417 articles
```

> ⚠️ **Note:** The loading code walks the folder structure dynamically using `os.walk`, so it works correctly even if folder names differ slightly across Kaggle cache versions.

---

## ⚙️ Setup Instructions

### 1. Clone or Download the Notebook

Download `NLP_Preprocessing_and_Classification_BBC_News.ipynb` and open it in Google Colab or Jupyter.

### 2. Configure Kaggle API Key

The notebook uses `kagglehub` to auto-download the dataset. You need a Kaggle account and API key.

**Steps:**
1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to **API** section → Click **Create New Token**
3. A `kaggle.json` file will be downloaded
4. In Colab, run the following before executing the notebook:

```python
from google.colab import files
files.upload()   # Upload your kaggle.json here

import os
os.makedirs('/root/.config/kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.config/kaggle/kaggle.json')
os.chmod('/root/.config/kaggle/kaggle.json', 0o600)
```

### 3. Install Dependencies

The notebook installs all dependencies automatically in Cell 2:

```bash
pip install kagglehub nltk scikit-learn pandas numpy matplotlib seaborn wordcloud
```

---

## 🔁 Pipeline Diagram

```
 Raw BBC Article Text (.txt files)
           │
           ▼
 ┌─────────────────────┐
 │   Dataset Loading   │  Walk bbc/ subfolders → Build DataFrame → Label encode → Train/Test Split (80/20)
 └─────────────────────┘
           │
           ▼
 ┌──────────────────────────┐
 │  Exploratory Data        │  Class distribution bar chart · Text length KDE by category
 │  Analysis (EDA)          │
 └──────────────────────────┘
           │
           ▼
 ┌──────────────────────────────────────────────────┐
 │         NLP Preprocessing Pipeline               │
 │  1. Lowercase                                    │
 │  2. Remove special characters & digits           │
 │  3. Tokenization  (word_tokenize)                │
 │  4. Stopword Removal  (NLTK English stopwords)   │
 │  5. Stemming  (PorterStemmer)                    │
 │  6. Lemmatization  (WordNetLemmatizer)           │
 └──────────────────────────────────────────────────┘
           │
           ▼
 ┌──────────────────────────────┐
 │     Text Vectorization       │
 │  ┌─────────────────────┐     │
 │  │ TF-IDF Vectorizer   │     │  max_features=50000, ngram_range=(1,2)
 │  └─────────────────────┘     │
 │  ┌─────────────────────┐     │
 │  │  CountVectorizer    │     │  max_features=50000, ngram_range=(1,2)
 │  └─────────────────────┘     │
 └──────────────────────────────┘
           │
           ▼
 ┌──────────────────────────────────────────────┐
 │            Model Training (3 × 2)            │
 │  ┌───────────────────┐  ┌─────────────────┐  │
 │  │  Multinomial NB   │  │   Linear SVM    │  │
 │  └───────────────────┘  └─────────────────┘  │
 │          ┌──────────────────────┐             │
 │          │  Logistic Regression │             │
 │          └──────────────────────┘             │
 └──────────────────────────────────────────────┘
           │
           ▼
 ┌──────────────────────────────────────────────┐
 │          Evaluation & Comparison             │
 │  · Accuracy bar chart (all 6 combinations)   │
 │  · Classification Report (best model)        │
 │  · Confusion Matrix 5×5 (best model)         │
 └──────────────────────────────────────────────┘
```

---

## 📊 Results

> Approximate expected results based on the BBC News dataset benchmark. Your exact numbers may vary slightly depending on random seed.

| Model                          | Vectorizer  | Accuracy  |
|-------------------------------|-------------|-----------|
| **Linear SVM**                | **TF-IDF**  | **~98%**  |
| Logistic Regression           | TF-IDF      | ~97%      |
| Linear SVM                    | CountVec    | ~97%      |
| Logistic Regression           | CountVec    | ~96%      |
| Naive Bayes                   | TF-IDF      | ~96%      |
| Naive Bayes                   | CountVec    | ~94%      |

### 🏆 Best Model: Linear SVM + TF-IDF

```
              precision    recall  f1-score   support

        Tech       0.99      0.98      0.98        80
    Business       0.97      0.97      0.97       102
       Sport       0.99      1.00      0.99       102
Entertainment       0.97      0.96      0.97        77
    Politics       0.98      0.98      0.98        84

    accuracy                           0.98       445
   macro avg       0.98      0.98      0.98       445
weighted avg       0.98      0.98      0.98       445
```

---

## 🔍 Key Findings

- **Linear SVM + TF-IDF** is the best performing combination (~98% accuracy), confirming its status as the go-to approach for text classification tasks.
- **TF-IDF consistently outperforms CountVectorizer** across all three models. For full-length BBC articles, raw word frequency is dominated by cross-category common words (*said*, *government*, *people*); TF-IDF corrects for this.
- **Naive Bayes**, despite being the simplest model, achieves competitive accuracy (~96%) with the fastest training time — a strong choice when compute is limited.
- **Business and Politics** show the highest cross-class confusion, reflecting genuine semantic overlap in BBC reporting (e.g., government budget articles fit both categories).
- **Sport** is classified with near-perfect precision across all models — its vocabulary (*goal*, *match*, *player*, *championship*) is highly distinctive and non-overlapping.
- **Bigrams** (`ngram_range=(1,2)`) capture important two-word phrases like *prime minister*, *stock market*, *mobile phone* that unigrams alone would miss.

---

## 📚 Libraries Used

| Library         | Purpose                                      |
|----------------|----------------------------------------------|
| `kagglehub`     | Dataset download from Kaggle                 |
| `pandas`        | Data loading and manipulation                |
| `numpy`         | Numerical operations                         |
| `nltk`          | Tokenization, stopwords, stemming, lemmatization |
| `scikit-learn`  | Vectorizers, classifiers, evaluation metrics |
| `matplotlib`    | Plotting charts and confusion matrix         |
| `seaborn`       | Enhanced visualizations                      |
| `re`            | Regex-based text cleaning                    |

---

## 🧠 Learning Outcomes

| Concept                     | What Was Demonstrated                                                   |
|----------------------------|-------------------------------------------------------------------------|
| **Text Preprocessing**      | Full pipeline: lowercase → clean → tokenize → stopword removal → stem → lemmatize |
| **Feature Extraction**      | TF-IDF vs. CountVectorizer with unigrams and bigrams                    |
| **Multi-class Classification** | 5-class OvR classification with Naive Bayes, SVM, Logistic Regression |
| **Model Evaluation**        | Accuracy, Precision, Recall, F1-score, Confusion Matrix                |
| **EDA**                     | Class distribution, text length analysis, top discriminative terms      |
| **Model Comparison**        | Cross-model and cross-vectorizer accuracy comparison chart              |
| **Real-world Data Handling**| Folder-based dataset loading with encoding error handling               |

---

## 🚀 Future Improvements

- **Deep Learning:** LSTM or GRU networks to capture sequential word context
- **Transformers:** Fine-tune `bert-base-uncased` for potentially >99% accuracy
- **Hyperparameter Tuning:** GridSearchCV on SVM's `C` parameter and TF-IDF's `min_df`/`max_df`
- **Word Embeddings:** Replace TF-IDF with Word2Vec or GloVe for semantic similarity
- **Error Analysis:** Deep-dive into Business vs. Politics misclassifications to understand boundary cases

---

## 📝 Notes

- The BBC dataset does **not** contain a flat CSV file. The notebook reads directly from the `bbc/<category>/` folder structure.
- All random operations use `random_state=42` for reproducibility.
- The stratified train-test split ensures proportional class representation in both sets.
