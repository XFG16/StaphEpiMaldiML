# This file was manually written after model_tuning.txt was created.
# Best classifiers were chosen based on AUROC performance in model_tuning.txt for each antibiotic.

from sklearn.svm import SVC
from lightgbm import LGBMClassifier


class BCOLORS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


SELECTED_ANTIBIOTICS = [
    "Ceftriaxone",
    "Ciprofloxacin",
    "Clindamycin",
    "Oxacillin",
    "Tetracycline",
    "Rifampicin",
    "Gentamicin",
    "Fusidic acid",
    "Amoxicillin-Clavulanic acid",
    "Cotrimoxazole",
    "Penicillin",
    "Tigecycline",
    "Ampicillin-Amoxicillin",
    "Teicoplanin",
]

BEST_CLASSIFIERS = {
    "Ceftriaxone": {
        "clf_name": "LightGBM",
        "clf": LGBMClassifier(),
        "params": {
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "n_estimators": 400,
            "objective": "binary",
            "verbose": -1,
        },
        "needs_scaled_data": False,
    },
    "Ciprofloxacin": {
        "clf_name": "LightGBM",
        "clf": LGBMClassifier(),
        "params": {
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "n_estimators": 400,
            "objective": "binary",
            "verbose": -1,
        },
        "needs_scaled_data": False,
    },
    "Clindamycin": {
        "clf_name": "Support Vector Machine",
        "clf": SVC(),
        "params": {"C": 1, "gamma": "auto", "kernel": "rbf", "probability": True},
        "needs_scaled_data": True,
    },
    "Oxacillin": {
        "clf_name": "Support Vector Machine",
        "clf": SVC(),
        "params": {"C": 10, "gamma": "auto", "kernel": "rbf", "probability": True},
        "needs_scaled_data": True,
    },
    "Tetracycline": {
        "clf_name": "LightGBM",
        "clf": LGBMClassifier(),
        "params": {
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "binary",
            "verbose": -1,
        },
        "needs_scaled_data": False,
    },
    "Rifampicin": {
        "clf_name": "LightGBM",
        "clf": LGBMClassifier(),
        "params": {
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "n_estimators": 400,
            "objective": "binary",
            "verbose": -1,
        },
        "needs_scaled_data": False,
    },
    "Gentamicin": {
        "clf_name": "LightGBM",
        "clf": LGBMClassifier(),
        "params": {
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "n_estimators": 400,
            "objective": "binary",
            "verbose": -1,
        },
        "needs_scaled_data": False,
    },
    "Fusidic acid": {
        "clf_name": "Support Vector Machine",
        "clf": SVC(),
        "params": {"C": 1, "gamma": "auto", "kernel": "rbf", "probability": True},
        "needs_scaled_data": True,
    },
    "Amoxicillin-Clavulanic acid": {
        "clf_name": "LightGBM",
        "clf": LGBMClassifier(),
        "params": {
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "n_estimators": 400,
            "objective": "binary",
            "verbose": -1,
        },
        "needs_scaled_data": False,
    },
    "Cotrimoxazole": {
        "clf_name": "LightGBM",
        "clf": LGBMClassifier(),
        "params": {
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "n_estimators": 400,
            "objective": "binary",
            "verbose": -1,
        },
        "needs_scaled_data": False,
    },
    "Penicillin": {
        "clf_name": "Support Vector Machine",
        "clf": SVC(),
        "params": {"C": 10, "gamma": "scale", "kernel": "rbf", "probability": True},
        "needs_scaled_data": True,
    },
    "Tigecycline": {
        "clf_name": "Support Vector Machine",
        "clf": SVC(),
        "params": {"C": 10, "gamma": "scale", "kernel": "rbf", "probability": True},
        "needs_scaled_data": True,
    },
    "Ampicillin-Amoxicillin": {
        "clf_name": "Support Vector Machine",
        "clf": SVC(),
        "params": {"C": 10, "gamma": "scale", "kernel": "rbf", "probability": True},
        "needs_scaled_data": True,
    },
    "Teicoplanin": {
        "clf_name": "Support Vector Machine",
        "clf": SVC(),
        "params": {"C": 10, "gamma": "scale", "kernel": "rbf", "probability": True},
        "needs_scaled_data": True,
    },
}