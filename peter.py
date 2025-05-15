import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

spam_data = pd.read_csv("spam.csv", encoding="latin1")

print(f"Shape after reading: {spam_data.shape}")

spam_data = spam_data.rename(columns={"v1": "label", "v2": "message"})
spam_data["label"] = spam_data["label"].map({"ham": 0, "spam": 1})

print(f"Shape after renaming and mapping: {spam_data.shape}")

print(spam_data[spam_data.isnull().any(axis=1)])

spam_data["message"].fillna("", inplace=True)

spam_data = spam_data.dropna(subset=["message"])

print(f"Shape after handling missing values: {spam_data.shape}")

X_train, X_test, y_train, y_test = train_test_split(spam_data["message"], spam_data["label"], test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
y_train_pred = model.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")
input_data="Hey, i'm going to the movies today"
input_tfidf = vectorizer.transform([input_data])
prediction = model.predict(input_tfidf)[0]
if(prediction==1):
    print("Spam")
else:
    print("Not Spam")
