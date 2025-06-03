# Enhanced Taxi Fare Prediction
# This script provides an improved workflow for loading the taxi fare dataset,
# performing exploratory data analysis, cleaning, feature engineering, and
# training a more robust model with cross validation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = Path("/kaggle/input/taxi-trip-fare-data-2023/Taxi_Trip_Data_preprocessed.csv")


def load_data(path: Path) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious outliers from the dataset."""
    df = df[df["duration"] > 30 / 60]
    df = df[df["fare_amount"] > 1]
    df = df[df["trip_distance"] > 0.02]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features used for modeling."""
    df = df.copy()
    # average speed in miles per hour
    df["avg_speed"] = df["trip_distance"] / (df["duration"] / 60)
    return df


def eda_plots(df: pd.DataFrame) -> None:
    """Generate basic exploratory plots."""
    sns.histplot(df["trip_distance"], bins=50, kde=True)
    plt.title("Trip Distance Distribution")
    plt.show()

    sns.histplot(df["fare_amount"], bins=50, kde=True, color="orange")
    plt.title("Fare Amount Distribution")
    plt.show()

    sns.scatterplot(x="trip_distance", y="fare_amount", data=df, alpha=0.2)
    sns.regplot(x="trip_distance", y="fare_amount", data=df, scatter=False, color="red")
    plt.title("Distance vs Fare")
    plt.show()

    corr = df.select_dtypes("number").corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


def train_models(df: pd.DataFrame):
    """Train linear regression and random forest models with grid search."""
    features = ["trip_distance", "duration", "avg_speed"]
    X = df[features]
    y = df["fare_amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    param_grid = [
        {"model": [LinearRegression()]},
        {
            "model": [RandomForestRegressor(random_state=42)],
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10],
        },
    ]

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Best model:", grid.best_params_)
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.3f}")

    return best_model


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_features(df)

    # Uncomment the line below to generate EDA plots when running interactively.
    # eda_plots(df)

    train_models(df)


if __name__ == "__main__":
    main()
