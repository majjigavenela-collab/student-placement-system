from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_FOLDER / "dataset" / "placement_data.csv"
MODEL_FOLDER = PROJECT_FOLDER / "model"
MODEL_FILE = MODEL_FOLDER / "placement_model.pkl"

FEATURE_COLUMNS = [
    "CGPA",
    "Major Projects",
    "Workshops/Certificatios",
    "Mini Projects",
    "Skills",
    "Communication Skill Rating",
    "Internship",
    "Hackathon",
    "12th Percentage",
    "10th Percentage",
    "backlogs",
]
NUMERIC_COLUMNS = [
    "CGPA",
    "Major Projects",
    "Workshops/Certificatios",
    "Mini Projects",
    "Skills",
    "Communication Skill Rating",
    "12th Percentage",
    "10th Percentage",
    "backlogs",
]
CATEGORICAL_COLUMNS = ["Internship", "Hackathon"]
TARGET_COLUMN = "PlacementStatus"


def main():
    # sep=None lets pandas detect whether this file uses commas or tabs.
    data = pd.read_csv(DATA_FILE, sep=None, engine="python")
    data.columns = data.columns.astype(str).str.strip()

    # Some CSV exports include an extra row-number column.
    data = data.loc[:, ~data.columns.str.match(r"^Unnamed")]
    data = data.drop_duplicates()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing these columns: " + ", ".join(missing_columns)
        )

    data = data[required_columns].copy()
    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN].astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)
    ).map({"placed": 1, "notplaced": 0})
    data = data.dropna(subset=[TARGET_COLUMN])

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(int)

    if y.nunique() != 2:
        raise ValueError("The target column must contain both Placed and NotPlaced rows.")

    numeric_preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessing, NUMERIC_COLUMNS),
            ("categorical", categorical_preprocessing, CATEGORICAL_COLUMNS),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    fitted_pipelines = {}
    print("Dataset loaded:", data.shape)
    print("Training rows:", len(X_train), "| Test rows:", len(X_test))

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessing", preprocessing), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        results[name] = metrics
        fitted_pipelines[name] = pipeline
        print(f"\n{name}")
        for metric, score in metrics.items():
            print(f"{metric.capitalize():10s}: {score:.4f}")

    best_name = max(results, key=lambda name: results[name]["f1"])
    best_pipeline = fitted_pipelines[best_name]

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_FILE)
    print(f"\nBest model by F1 score: {best_name}")
    print("Saved complete preprocessing + model to:", MODEL_FILE)


if __name__ == "__main__":
    main()
