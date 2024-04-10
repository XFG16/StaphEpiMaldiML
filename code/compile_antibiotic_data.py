# FILE 1

# Summary statistics for the following antibiotics for S. epidermidis in DRIAMS-A

import numpy as np
import pandas as pd

BIN_INDICES = [f"bin_index_{i}" for i in range(6000)]

raw_meta_datas = []

for i in range(2015, 2019):
    temp_meta_data = pd.read_csv(f"./data/antibiotics/DRIAMS-A/id/{i}/{i}_clean.csv")
    temp_meta_data_copy = temp_meta_data.copy()
    temp_meta_data_copy = temp_meta_data_copy[
        temp_meta_data_copy["species"] == "Staphylococcus epidermidis"
    ]
    raw_meta_datas.append(temp_meta_data_copy)

antibiotics = set()
for current in raw_meta_datas:
    columns_list = current.columns.tolist()
    antibiotics.update(columns_list)

ignored_columns = [
    "code",
    "species",
    "laboratory_species",
    "Gentamicin_high_level",
    "Meropenem_without_meningitis",
    "Meropenem_with_meningitis",
    "Isoniazid_.1mg-l",
    "Rifampicin_1mg-l",
    "Ethambutol_5mg-l",
    "Vancomycin_GRD",
    "Teicoplanin_GRD",
    "Cefoxitin_screen",
    "Penicillin_with_endokarditis",
    "Penicillin_without_meningitis",
    "Penicillin_without_endokarditis",
    "Penicillin_with_pneumonia",
    "Penicillin_with_other_infections",
    "Penicillin_with_meningitis",
    "Meropenem_with_pneumonia",
    "Amoxicillin-Clavulanic acid_uncomplicated_HWI",
    "Isoniazid_.4mg-l",
    "Strepomycin_high_level",
]
for ignored in ignored_columns:
    antibiotics.discard(ignored)

# See Tables 1 and 2 of manuscript for clarification

special_cases = [
    "Cefepime",
    "Meropenem",
    "Piperacillin-Tazobactam",
    "Imipenem",
    "Cefuroxime",
    "Cefazolin",
]

for antibiotic in antibiotics:
    if antibiotic in special_cases:
        print(f"{antibiotic} is a special case and will be skipped.")
        continue

    model_data = []
    model_labels = []

    print(f"Currently on {antibiotic}.")
    for i, current_data in enumerate(raw_meta_datas):
        if not antibiotic in current_data.columns:
            continue

        current_data = current_data[["code", antibiotic]]
        current_data = current_data[current_data[antibiotic].isin(["I", "R", "S"])]
        current_data.replace("I", "R")

        for file_name in current_data["code"]:
            with open(
                f"./data/antibiotics/DRIAMS-A/binned_6000/{i + 2015}/{file_name}.txt"
            ) as file:
                lines = file.readlines()

            bin_data = [
                float(line.strip().split()[1]) for j, line in enumerate(lines) if j > 0
            ]
            label = (
                1
                if current_data.loc[current_data["code"] == file_name, antibiotic].iloc[
                    0
                ]
                == "R"
                else 0
            )

            model_data.append(bin_data)
            model_labels.append(label)

    model_data = pd.DataFrame(
        model_data, columns=BIN_INDICES
    )
    model_labels = pd.DataFrame(model_labels, columns=["label"])

    model_size = len(model_labels)
    if model_size == 0:
        print(f"Invalid model size for {antibiotic}")
        continue

    if (model_labels["label"] == 0).sum() <= 5 or (
        model_labels["label"] == 1
    ).sum() <= 5:
        print(f"Too little positive or negative cases for {antibiotic}")
        continue

    positive_class_prevalence = (model_labels["label"] == 1).sum() / len(model_labels)
    print(
        f"Data compilation for {antibiotic} complete. Model size is {model_size} and positive class prevalence is {positive_class_prevalence}."
    )

    with open("compiled_antibiotic_data_log.csv", "a") as log:
        log.write(f"\"{antibiotic}\",{model_size},{positive_class_prevalence}\n")
    
    model_overall = model_data.assign(label=model_labels)
    if (len(model_overall) != len(model_labels)):
        print("Error in combining dataframes.")
        break
    model_overall.to_csv(f"compiled_data/{antibiotic}.csv", index=False)
