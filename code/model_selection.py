# FILE 3

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from constants import SELECTED_ANTIBIOTICS

classifiers = {
    "Random Forest": RandomForestClassifier(random_state=0, n_jobs=12),
    "Logistic Regression": LogisticRegression(random_state=0, n_jobs=12),
    "Support Vector Machine": SVC(probability=True, random_state=0),
    "Naive Bayes": GaussianNB(),
    "LightGBM": LGBMClassifier(random_state=0, n_jobs=12, verbose=-1),
    "Multilayer Perceptron": MLPClassifier(
        hidden_layer_sizes=(256, 128),
        random_state=0,
        max_iter=500,
    ),
}

for antibiotic in SELECTED_ANTIBIOTICS:
    antibiotic_data = pd.read_csv(f"feature_selected_data/{antibiotic}.csv")
    model_data = antibiotic_data.drop("label", axis=1)
    model_labels = antibiotic_data[["label"]]

    if antibiotic == "Logistic Regression" or antibiotic == "Support Vector Machine":
        scaler = StandardScaler()
        scaler.fit(model_data)
        model_data = scaler.transform(model_data)

    X_train, X_test, y_train, y_test = train_test_split(
        model_data, model_labels, test_size=0.3, random_state=0, stratify=model_labels
    )

    print(f"Currently on {antibiotic}.")

    for name, clf in classifiers.items():
        clf.fit(X_train, np.ravel(y_train))

        if hasattr(clf, "predict_proba"):
            y_scores = clf.predict_proba(X_test)[:, 1]
        else:
            y_scores = clf.decision_function(X_test)
        auroc = roc_auc_score(np.ravel(y_test), y_scores)

        print(f"Antibiotic: {antibiotic}, Model: {name}, AUROC: {auroc:.4f}")
        with open("model_selection.csv", "a") as log:
            log.write(f'"{antibiotic}","{name}",{auroc:.4f}\n')
