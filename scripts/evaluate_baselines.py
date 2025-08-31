# scripts/evaluate_baselines.py
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

SPLITS = {
    "random": ("data/splits/random_train.parquet", "data/splits/random_test.parquet"),
    "temporal": ("data/splits/temporal_train.parquet", "data/splits/temporal_test.parquet"),
    "hospital": ("data/splits/hospital_train.parquet", "data/splits/hospital_test.parquet"),
}

TARGET = "hospital_mortality"

def load_data(train_path, test_path):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    # drop any string/categorical cols (like diagnosisstring, gender)
    X_train = X_train.select_dtypes(include="number").fillna(0)
    X_test = X_test.select_dtypes(include="number").fillna(0)

    return X_train, y_train, X_test, y_test


def evaluate_split(name, train_path, test_path):
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds > 0.5)
    auc = roc_auc_score(y_test, preds)

    return {"split": name, "accuracy": acc, "auroc": auc}


def main():
    results = []
    for name, (train_path, test_path) in SPLITS.items():
        print(f"Evaluating {name} split …")
        res = evaluate_split(name, train_path, test_path)
        results.append(res)

    df = pd.DataFrame(results)
    print("\n=== Baseline Results ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
