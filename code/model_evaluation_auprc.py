# FILE 6

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from constants import BEST_CLASSIFIERS, BCOLORS

compiled_data_descriptions = pd.read_csv("compiled_antibiotic_data_log.csv")

for antibiotic, test in BEST_CLASSIFIERS.items():
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

    current_model_data = model_data.copy()
    if test["needs_scaled_data"]:
        scaler = StandardScaler()
        scaler.fit(current_model_data)
        current_model_data = scaler.transform(current_model_data)

    scores = []
    best_score = 0.0
    for i in range(10):
        print(f"Iteration {i + 1} of {antibiotic}...")

        clf = test["clf"]
        clf.set_params(**test["params"])
        if test["clf_name"] != "Support Vector Machine":
            clf.set_params(n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(
            current_model_data, model_labels, test_size=0.3, stratify=model_labels
        )
        clf.fit(X_train, y_train)

        y_pred_proba = clf.predict_proba(X_test)[:, 1] 
        score = average_precision_score(y_test, y_pred_proba)

        scores.append(score)
        best_score = max(best_score, score)

    mean = np.mean(scores)
    std = np.std(scores)

    if mean > 0.7:
        print(
            BCOLORS.OKGREEN
            + f"Antibiotic: {antibiotic}, Model: {test['clf_name']}, Best AUPRC: {best_score}, Mean AUPRC: {mean}, AUPRC STD: {std}\n"
            + BCOLORS.ENDC
        )
    else:
        print(
            f"Antibiotic: {antibiotic}, Model: {test['clf_name']}, Best AUPRC: {best_score}, Mean AUPRC: {mean}, AUPRC STD: {std}\n"
        )
    with open("model_evaluation_auprc.txt", "a") as text_file:
        text_file.write(
            f"Antibiotic: {antibiotic}, Model: {test['clf_name']}, Best AUPRC: {best_score}, Mean AUPRC: {mean}, AUPRC STD: {std}\n"
        )
