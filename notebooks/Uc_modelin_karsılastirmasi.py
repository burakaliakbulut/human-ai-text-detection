import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv(r"D:\Masaüstü\human_ai\data\processed\ai_human_6000_final.csv")

print(df.shape)
print(df["label"].value_counts())


X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=100000,
    min_df=5
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

RANDOM_STATE = 42

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        random_state=RANDOM_STATE
        ),
    "Linear SVM": CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE),
        method="sigmoid",
        cv=5
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


results = []

for name, model in models.items():
    print("\n" + "="*50)
    print(f"MODEL: {name}")
    print("="*50)

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": acc
    })


results_df = pd.DataFrame(results)
print("\nMODEL KARŞILAŞTIRMASI")
print(results_df)


results_df.to_csv("model_results.csv", index=False, encoding="utf-8-sig")

import joblib

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(models["Logistic Regression"], "logreg.pkl")
joblib.dump(models["Linear SVM"], "svm.pkl")
joblib.dump(models["Random Forest"], "rf.pkl")
