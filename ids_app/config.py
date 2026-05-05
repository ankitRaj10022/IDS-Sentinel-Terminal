from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

CLASSICAL_TRAIN_DATA = ROOT_DIR / "kddtrain.csv"
CLASSICAL_TEST_DATA = ROOT_DIR / "kddtest.csv"
DNN_TRAIN_DATA = ROOT_DIR / "dnn" / "kdd" / "binary" / "Training.csv"
DNN_TEST_DATA = ROOT_DIR / "dnn" / "kdd" / "binary" / "Testing.csv"

FRONTEND_DIR = ROOT_DIR / "frontend"
REPORTS_DIR = ROOT_DIR / "Reports"

AUTOMATION_DIR = ROOT_DIR / "automation"
JOBS_DIR = AUTOMATION_DIR / "jobs"
RUNS_DIR = AUTOMATION_DIR / "runs"
LEGACY_DIR = AUTOMATION_DIR / "legacy"

CLASSICAL_MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Gaussian Naive Bayes",
    "decision_tree": "Decision Tree",
    "adaboost": "AdaBoost",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors",
}

LEGACY_CLASSICAL_FILES = {
    "legacy_lr": ("Logistic Regression", ROOT_DIR / "classical" / "predictedlabelLR.txt"),
    "legacy_nb": ("Gaussian Naive Bayes", ROOT_DIR / "classical" / "predictedlabelNB.txt"),
    "legacy_knn": ("K-Nearest Neighbors", ROOT_DIR / "classical" / "predictedlabelKNN.txt"),
    "legacy_dt": ("Decision Tree", ROOT_DIR / "classical" / "predictedlabelDT.txt"),
    "legacy_ab": ("AdaBoost", ROOT_DIR / "classical" / "predictedlabelAB.txt"),
    "legacy_rf": ("Random Forest", ROOT_DIR / "classical" / "predictedlabelRF.txt"),
    "legacy_svm_linear": ("SVM Linear", ROOT_DIR / "classical" / "predictedlabelSVM-linear.txt"),
    "legacy_svm_rbf": ("SVM RBF", ROOT_DIR / "classical" / "predictedlabelSVM-rbf.txt"),
}

LEGACY_DNN_FILES = {
    "legacy_dnn1": ("DNN 1 Layer", ROOT_DIR / "dnn" / "dnnres" / "dnn1predicted.txt"),
    "legacy_dnn2": ("DNN 2 Layer", ROOT_DIR / "dnn" / "dnnres" / "dnn2predicted.txt"),
    "legacy_dnn3": ("DNN 3 Layer", ROOT_DIR / "dnn" / "dnnres" / "dnn3predicted.txt"),
    "legacy_dnn4": ("DNN 4 Layer", ROOT_DIR / "dnn" / "dnnres" / "dnn4predicted.txt"),
    "legacy_dnn5": ("DNN 5 Layer", ROOT_DIR / "dnn" / "dnnres" / "dnn5predicted.txt"),
}

DNN_LAYER_SPECS = {
    1: [1024],
    2: [1024, 768],
    3: [1024, 768, 512],
    4: [1024, 768, 512, 256],
    5: [1024, 768, 512, 256, 128],
}

CLASSICAL_PROFILES = {
    "fast": {
        "models": ["logistic_regression", "naive_bayes", "decision_tree", "adaboost", "random_forest"],
        "train_sample": 40000,
        "test_sample": 15000,
    },
    "balanced": {
        "models": ["logistic_regression", "naive_bayes", "decision_tree", "adaboost", "random_forest", "knn"],
        "train_sample": 100000,
        "test_sample": 30000,
    },
    "full": {
        "models": ["logistic_regression", "naive_bayes", "decision_tree", "adaboost", "random_forest"],
        "train_sample": None,
        "test_sample": None,
    },
}

DNN_PROFILES = {
    "fast": {
        "architectures": [1, 3],
        "train_sample": 50000,
        "test_sample": 15000,
        "epochs": 3,
        "batch_size": 128,
    },
    "balanced": {
        "architectures": [1, 3, 5],
        "train_sample": 120000,
        "test_sample": 40000,
        "epochs": 5,
        "batch_size": 128,
    },
    "full": {
        "architectures": [1, 2, 3, 4, 5],
        "train_sample": None,
        "test_sample": None,
        "epochs": 10,
        "batch_size": 128,
    },
}
