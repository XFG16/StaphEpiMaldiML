# FILE 4

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from constants import SELECTED_ANTIBIOTICS, BCOLORS

classifier_tests = {
    "Random Forest": {
        "clf": RandomForestClassifier(),
        "param_grid": {
            "random_state": [0],
            "n_estimators": [100, 200, 400],
            "max_features": ["sqrt", "log2", 60, 100],
        },
        "needs_scaled_data": False,
    },
    "Logistic Regression": {
        "clf": LogisticRegression(),
        "param_grid": {
            "random_state": [0],
            "C": [0.001, 0.01, 0.1],
            "penalty": ["l1", "l2"],
            "solver": ["lbfgs", "saga", "liblinear", "newton-cg"],
            "max_iter": [1000]
        },
        "needs_scaled_data": True,
    },
    "Support Vector Machine": {
        "clf": SVC(),
        "param_grid": {
            "random_state": [0],
            "probability": [True],
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto"],
        },
        "needs_scaled_data": True,
    },
    "Naive Bayes": {
        "clf": GaussianNB(),
        "param_grid": {},
        "needs_scaled_data": False,
    },
    "LightGBM": {
        "clf": LGBMClassifier(),
        "param_grid": {
            "random_state": [0],
            "verbose": [-1],
            "objective": ["binary"],
            "boosting_type": ["gbdt"],
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.1, 1],
        },
        "needs_scaled_data": False,
    },
    "Multilayer Perceptron": {
        "clf": MLPClassifier(),
        "param_grid": {
            "random_state": [0],
            "hidden_layer_sizes": [
                (512, 256, 128),
                (512, 128, 64),
                (256, 64),
                (256, 128),
            ],
            "activation": ["relu"],
            "alpha": [0.0001],
            "max_iter": [1000],
        },
        "needs_scaled_data": False,
    },
}

compiled_data_descriptions = pd.read_csv("compiled_antibiotic_data_log.csv")

for antibiotic in SELECTED_ANTIBIOTICS:
    antibiotic_data = pd.read_csv(f"feature_selected_data/{antibiotic}.csv")
    model_data = antibiotic_data.drop("label", axis=1)
    model_labels = np.ravel(antibiotic_data[["label"]])

    model_size = compiled_data_descriptions.loc[
        compiled_data_descriptions["Antibiotic"] == antibiotic, "Model Size"
    ].values[0]
    positive_class_prevalence = compiled_data_descriptions.loc[
        compiled_data_descriptions["Antibiotic"] == antibiotic,
        "Positive Class Prevalence",
    ].values[0]
    print(
        f"Currently on {antibiotic}. Model size is {model_size}, number of columns is {len(model_data.columns)}, and positive class prevalence is {positive_class_prevalence:.4f}."
    )

    for name, test in classifier_tests.items():
        current_model_data = model_data.copy()
        if test["needs_scaled_data"]:
            scaler = StandardScaler()
            scaler.fit(current_model_data)
            current_model_data = scaler.transform(current_model_data)

        try:
            clf = test["clf"]
            grid = GridSearchCV(
                estimator=clf,
                param_grid=test["param_grid"],
                cv=5,
                scoring="roc_auc",
                verbose=1,
                n_jobs=-1,
            )
            grid.fit(current_model_data, model_labels)

            if grid.best_score_ > 0.7:
                print(
                    BCOLORS.OKGREEN
                    + f"Antibiotic: {antibiotic}, Model: {name}, AUROC: {grid.best_score_}, Model Size: {model_size}, Positive Class Prevalence: {positive_class_prevalence:.4f}, Params: {grid.best_params_}\n"
                    + BCOLORS.ENDC
                )
            else:
                print(
                    f"Antibiotic: {antibiotic}, Model: {name}, AUROC: {grid.best_score_}, Model Size: {model_size}, Positive Class Prevalence: {positive_class_prevalence:.4f}, Params: {grid.best_params_}\n"
                )
            with open("model_tuning.txt", "a") as text_file:
                text_file.write(
                    f"Antibiotic: {antibiotic}, Model: {name}, AUROC: {grid.best_score_}, Model Size: {model_size}, Positive Class Prevalence: {positive_class_prevalence:.4f}, Params: {grid.best_params_}\n"
                )
        except Exception as e:
            print(
                BCOLORS.FAIL + f"[ERROR] {e}, Antibiotic: {antibiotic}" + BCOLORS.ENDC
            )

