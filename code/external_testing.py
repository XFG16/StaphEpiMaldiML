# FILE 8

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from constants import SELECTED_ANTIBIOTICS, BEST_CLASSIFIERS, BCOLORS

BIN_INDICES = [f"bin_index_{i}" for i in range(6000)]

compiled_data_descriptions = pd.read_csv("compiled_antibiotic_data_log.csv")

external_sites = ["DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]
external_site_data = {}
external_site_labels = {}

for site in external_sites:
    meta_data = pd.read_csv(f"./data/antibiotics/{site}/id/2018/2018_clean.csv")
    meta_data = meta_data[meta_data["species"] == "Staphylococcus epidermidis"]

    print(f"Currently on {site}. Size is {len(meta_data.index)}")

    site_data = []
    site_labels = {}

    failed_opened = 0

    for file_name in meta_data["code"]:
        try:
            with open(
                f"./data/antibiotics/{site}/binned_6000/2018/{file_name}.txt"
            ) as file:
                lines = file.readlines()
        except Exception as e:
            print(BCOLORS.FAIL + f"Failed to open {file_name} in {site}." + BCOLORS.ENDC)
            failed_opened += 1

        bin_data = [
            float(line.strip().split()[1]) for j, line in enumerate(lines) if j > 0
        ]

        site_data.append(bin_data)
        for antibiotic in SELECTED_ANTIBIOTICS:
            if not antibiotic in meta_data:
                continue

            label = (
                1
                if meta_data.loc[meta_data["code"] == file_name, antibiotic].iloc[
                    0
                ]
                == "R"
                else 0
            )

            if antibiotic in site_labels:
                site_labels[antibiotic].append(label)
            else:
                site_labels[antibiotic] = [label]
    
    external_site_data[site] = pd.DataFrame(site_data, columns=BIN_INDICES)
    external_site_labels[site] = site_labels

    print(f"Failed to open {failed_opened} files in {site}.")

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

    clf = test["clf"]
    clf.set_params(**test["params"])
    if test["clf_name"] != "Support Vector Machine":
        clf.set_params(n_jobs=-1)

    clf.fit(current_model_data, model_labels)

    selected_features = model_data.columns.tolist()

    for site in external_sites:
        if not antibiotic in external_site_labels[site]:
            continue

        model_data = external_site_data[site]
        model_data = model_data[selected_features]
        model_labels = pd.DataFrame(external_site_labels[site][antibiotic], columns=["label"])

        current_pcp = (model_labels["label"] == 1).sum() / len(model_labels)

        if (current_pcp == 0):
            with open("external_testing.txt", "a") as log:
                log.write(f"Antibiotic: {antibiotic}, Site: {site}, AUROC: NaN\n")
            print(BCOLORS.WARNING + f"Positive class prevalence for {antibiotic} in {site} is 0. No AUROC will be evaluated." + BCOLORS.ENDC)
            continue
        else:
            print(f"Positive class prevalence for {antibiotic} in {site} is {current_pcp}.")

        y_pred_proba = clf.predict_proba(model_data)[:, 1]
        score = roc_auc_score(np.ravel(model_labels), y_pred_proba)

        if score >= 0.7:
            print(BCOLORS.OKGREEN + f"Antibiotic: {antibiotic}, Site: {site}, AUROC: {score}" + BCOLORS.ENDC)
        elif score <= 0.5:
            print(BCOLORS.FAIL + f"Antibiotic: {antibiotic}, Site: {site}, AUROC: {score}" + BCOLORS.ENDC)
        else:
            print(f"Antibiotic: {antibiotic}, Site: {site}, AUROC: {score}")
        
        with open("external_testing.txt", "a") as log:
            log.write(f"Antibiotic: {antibiotic}, Site: {site}, AUROC: {score}\n")
    
