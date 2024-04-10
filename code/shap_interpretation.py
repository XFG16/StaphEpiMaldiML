# FILE 7

import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split

from constants import BEST_CLASSIFIERS

ANTIBIOTIC = "Ciprofloxacin"

test = BEST_CLASSIFIERS[ANTIBIOTIC]

clf = test["clf"]
clf.set_params(**test["params"], n_jobs=-1)

antibiotic_data = pd.read_csv(f"feature_selected_data/{ANTIBIOTIC}.csv")
model_data = antibiotic_data.drop("label", axis=1)
model_labels = np.ravel(antibiotic_data[["label"]])

X_train, X_test, y_train, y_test = train_test_split(
    model_data, model_labels, test_size=0.3, stratify=model_labels, random_state=0
)
clf.fit(X_train, y_train)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, max_display=5)