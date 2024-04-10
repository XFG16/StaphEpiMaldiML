# FILE 2

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from constants import SELECTED_ANTIBIOTICS

for antibiotic in SELECTED_ANTIBIOTICS:
    antibiotic_data = pd.read_csv(f"compiled_data/{antibiotic}.csv")
    model_data = antibiotic_data.drop("label", axis=1)
    model_labels = antibiotic_data[["label"]]
    print(f"Original feature size for {antibiotic}: {len(model_data.columns)}")

    clf = RandomForestClassifier(n_jobs=12, random_state=0)
    selector = SelectFromModel(estimator=clf, threshold="1.25*mean")
    selector.fit(model_data, np.ravel(model_labels))
    
    feature_importances = selector.estimator_.feature_importances_
    mean_feature_importance = np.mean(feature_importances)
    threshold_value = selector.threshold_
    
    print(f"Mean feature importance for {antibiotic}: {mean_feature_importance}")
    print(f"Threshold for feature selection: {threshold_value}")

    model_data = model_data.loc[:, selector.get_support()]

    model_overall = model_data.assign(label=model_labels)
    print(f"New feature size for {antibiotic}: {len(model_data.columns)}")
    model_overall.to_csv(f"feature_selected_data/{antibiotic}.csv", index=False)
