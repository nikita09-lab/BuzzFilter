# Buzz Filter

file structure

        Spam-Classifier/
        │
        ├── data/
        │   ├── raw/
        │   │   └── spam.csv
        │   │
        │   └── processed/
        │       └── cleaned_spam.csv
        │
        ├── notebooks/
        │   ├── 01_data_preprocessing.ipynb
        │   ├── 02_eda.ipynb
        │   └── 03_model_training.ipynb
        │
        ├── src/
        │   ├── preprocessing.py
        │   ├── feature_engineering.py
        │   ├── train_model.py
        │   ├── evaluate_model.py
        │   └── predict.py
        │
        ├── models/
        │   ├── spam_model.pkl
        │   └── tfidf_vectorizer.pkl
        │
        ├── app/
        │   ├── app.py
        │   └── utils.py
        │
        ├── reports/
        │   ├── eda_plots/
        │   │   ├── spam_ham_distribution.png
        │   │   ├── word_frequency.png
        │   │   └── message_length_distribution.png
        │   │
        │   └── model_comparison.csv
        │
        ├── tests/
        │   └── test_prediction.py
        │
        ├── requirements.txt
        ├── README.md
        ├── .gitignore
        └── LICENSE

To start Frontend
python -m streamlit run app/app.py