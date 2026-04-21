# train_model.py - CSV Version (9 Real Features)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')


def train_model():
    print("Training House Price Prediction Model...")
    print("="*60)

    os.makedirs('models', exist_ok=True)

    # Load CSV
    df = pd.read_csv("maharashtra_housing_data.csv")
    print(f"Loaded {len(df)} records from CSV")

    # Rename for consistency (optional but clean)
    df = df.rename(columns={"property_age": "age"})

    # Encode categorical features
    le_location = LabelEncoder()
    le_furnishing = LabelEncoder()
    le_ac = LabelEncoder()
    le_mainroad = LabelEncoder()

    df["location_encoded"] = le_location.fit_transform(df["location"])
    df["furnishing_encoded"] = le_furnishing.fit_transform(df["furnishing"])
    df["ac_encoded"] = le_ac.fit_transform(df["air_conditioning"])
    df["mainroad_encoded"] = le_mainroad.fit_transform(df["main_road"])

    # Use ALL 9 real features
    feature_cols = [
        "area",
        "bedrooms",
        "bathrooms",
        "age",
        "parking",
        "location_encoded",
        "furnishing_encoded",
        "ac_encoded",
        "mainroad_encoded"
    ]

    X = df[feature_cols]
    y = df["price"]

    print(f"\nFeatures Used: {feature_cols}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    print("\nTraining...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Evaluation
    r2 = model.score(X_test_scaled, y_test)
    mae = mean_absolute_error(y_test, model.predict(X_test_scaled))

    print(f"\nR² Score: {r2:.4f}")
    print(f"MAE: ₹{mae:,.0f}")

    # Save model artifacts
    joblib.dump(model, "models/house_price_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_columns.pkl")

    joblib.dump(le_location, "models/location_encoder.pkl")
    joblib.dump(le_furnishing, "models/furnishing_encoder.pkl")
    joblib.dump(le_ac, "models/ac_encoder.pkl")
    joblib.dump(le_mainroad, "models/mainroad_encoder.pkl")

    # Metadata
    meta = {
    "features": feature_cols,
    "locations": list(le_location.classes_),   # ADD THIS
    "furnishing": list(le_furnishing.classes_), # ADD
    "ac": list(le_ac.classes_),                # ADD
    "main_road": list(le_mainroad.classes_),   # ADD
    "r2_score": float(r2),
    "mae": float(mae)
}
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nModel trained & saved successfully!")
    print("Run: python app.py")


if __name__ == "__main__":
    train_model()