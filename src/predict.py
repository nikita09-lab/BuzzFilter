import pickle

with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

message = ["Free iPhone! Click now"]

vector = vectorizer.transform(message)

print(model.predict(vector))