from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import pickle

df = pd.read_csv("data/cleaned/sms_spam_cleaned.csv")

df = df.dropna(subset=["clean_text"])

X = df["clean_text"]
y = df["label_num"]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)

model = LinearSVC()
model.fit(X, y)

with open("models/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)