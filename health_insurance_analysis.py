"""Predict healthcare costs with an interpretable Python regression workflow."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "insurance.csv"


def adjusted_r2(y_true, y_pred, predictors):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - predictors - 1)


def make_features(data):
    features = pd.get_dummies(data.drop(columns="charges", errors="ignore"), drop_first=True, dtype=float)
    features["age_sq"] = features["age"] ** 2
    features["bmi_sq"] = features["bmi"] ** 2
    for column in ["sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]:
        if column not in features:
            features[column] = 0.0
    features["age_children"] = features["age"] * features["children"]
    features["age_smoker"] = features["age"] * features["smoker_yes"]
    features["bmi_smoker"] = features["bmi"] * features["smoker_yes"]
    features["sex_smoker"] = features["sex_male"] * features["smoker_yes"]
    features["children_smoker"] = features["children"] * features["smoker_yes"]
    return features


def main():
    data = pd.read_csv(DATA).dropna().drop_duplicates()
    X = make_features(data)
    y = np.log(data["charges"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=999
    )

    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Rows: {len(data):,}")
    print(f"Test RMSE on log charges: {mean_squared_error(y_test, predictions) ** 0.5:.3f}")
    print(f"Test R2 on log charges: {r2_score(y_test, predictions):.3f}")
    train_predictions = model.predict(X_train)
    print(f"Training adjusted R2: {adjusted_r2(y_train, train_predictions, X.shape[1]):.3f}")

    example = pd.DataFrame(
        [{"age": 27, "sex": "male", "bmi": 28, "children": 0, "smoker": "no", "region": "southwest"}]
    )
    example_features = make_features(example).reindex(columns=X.columns, fill_value=0)
    print(f"Example predicted charge: ${np.exp(model.predict(example_features)[0]):,.0f}")


if __name__ == "__main__":
    main()
