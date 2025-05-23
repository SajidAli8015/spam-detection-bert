# 📧 Email Spam Detection Using Classical ML & BERT (DistilBERT)

## 📌 Overview

The objective of this project is to develop a **general-purpose spam detection model** that can classify text messages and emails as either "spam" or "ham" (not spam). We approached this problem in two phases:

1. **Start simple**: We began with a well-known SMS spam dataset and implemented classical machine learning models. These models serve as a strong **baseline** due to their simplicity, efficiency, and interpretability.
2. **Advance with BERT**: We then leveraged the **DistilBERT transformer model** to enhance spam detection, especially in the context of emails. After training on SMS data, we fine-tuned the model on a phishing email dataset to evaluate its generalizability and robustness.

The project aims to compare classical and modern NLP approaches in terms of accuracy and adaptability across different types of text data.

---

## 📁 Project Structure

```
spam-detection-project/
├── data/                     # Contains SMS and phishing email datasets
│   ├── spam.csv
│   └── phishing_email.csv
├── models/                  # Saved fine-tuned BERT models
│   ├── bert_sms_model/
│   └── bert_email_model/
├── notebooks/               # Cleaned Jupyter notebooks
│   ├── Classical_models.ipynb           # Logistic Regression & Naive Bayes
│   ├── bert_sms_model.ipynb            # DistilBERT trained on SMS
│   └── bert_email_finetune.ipynb       # DistilBERT fine-tuned on emails
├── requirements.txt         # Required Python libraries
├── .gitignore               # Files/folders to ignore in git
└── README.md                # Project overview and usage instructions
```

---

## 📦 Datasets Used

1. **SMS Spam Collection Dataset**

   * \~5,500 labelled messages as `spam` or `ham`
   * Used for both classical models and initial BERT training

2. **Phishing Email Dataset**

   * \~33,000 emails labeled as spam or ham
   * Contains real phishing and legitimate emails
   * Used for evaluating generalization and fine-tuning BERT

---

## 🧠 Models Implemented

### 1. Logistic Regression (TF-IDF)

* Text transformed using TF-IDF vectorization
* Optimized using `GridSearchCV`

### 2. Naive Bayes (TF-IDF)

* MultinomialNB on TF-IDF features
* Simple yet surprisingly strong for spam detection

### 3. BERT (DistilBERT)

* Fine-tuned pre-trained `distilbert-base-uncased`
* Used Hugging Face `Trainer` API
* Trained first on SMS, then fine-tuned on phishing email data

---

## 📊 Model Performance

| Model                   | Dataset          | Precision | Recall | F1 Score |
| ----------------------- | ---------------- | --------- | ------ | -------- |
| Logistic Regression     | SMS (Test)       | \~84%     | \~83%  | \~83%    |
| Naive Bayes             | SMS (Test)       | \~82%     | \~81%  | \~81%    |
| BERT (DistilBERT)       | SMS (Test)       | \~92%     | \~91%  | \~91%    |
| BERT (Trained on SMS)   | Email (Phishing) | \~78%     | \~75%  | \~76%    |
| BERT (Fine-tuned Email) | Email (Phishing) | \~91%     | \~90%  | \~90%    |

✅ **BERT significantly outperformed classical models, especially after fine-tuning on real phishing emails.**

---

## 🚀 Run Model Notebooks

* `notebooks/Classical_models.ipynb` – Logistic Regression and Naive Bayes on SMS
* `notebooks/bert_sms_model.ipynb` – BERT trained on SMS spam
* `notebooks/bert_email_finetune.ipynb` – Fine-tuned BERT on phishing emails

---

## 🛠️ Future Enhancements

* 📲 **Streamlit App** for interactive spam detection UI
* 🚀 **FastAPI Backend** to serve predictions
* 🔎 Explore **Explainable AI** for model interpretation
* 📈 Hyperparameter tuning using Optuna or Ray Tune

---

## 🧪 Dependencies

* Python ≥ 3.8
* `transformers`, `sklearn`, `torch`, `pandas`, `seaborn`, `matplotlib`, `nltk`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🙌 Credits

* [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
* [Phishing Email Dataset on Kaggle](https://www.kaggle.com/datasets/sureshkesireddy/phishing-emails)
* Hugging Face Transformers
* Scikit-learn

---

## 📬 Contact

For questions or suggestions, feel free to reach out!

> 📧 Sajid Ali · Data Scientist · alisajid8030@gmail.com
