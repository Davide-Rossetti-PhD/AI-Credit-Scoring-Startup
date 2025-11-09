# ==========================================
# Generate WeTrust Dataset
# ==========================================

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_wetrust_dataset(n):
    """
    Generate a synthetic dataset inspired by the WeTrust credit scoring model.
    Includes behavioral, responsibility, and financial features.
    """

    data = {
        # TIME / FAMILY CONNECTION features
        "call_frequency": np.clip(np.random.normal(5, 2, n), 0, 15),   # calls per week
        "call_duration": np.clip(np.random.normal(6, 3, n), 0, 25),    # avg minutes per call
        "msg_count": np.clip(np.random.normal(18, 10, n), 0, 80),      # messages per day

        # RESPONSIBILITY features
        "commute_consistency": np.clip(np.random.beta(5, 2, n), 0, 1),  # ratio of days with routine
        "night_out_ratio": np.clip(np.random.beta(2, 6, n), 0, 1),      # ratio of random nights out

        # APP USAGE features
        "app_betting": np.random.choice([0, 1], p=[0.85, 0.15], size=n),
        "app_learning_news": np.random.choice([0, 1], p=[0.45, 0.55], size=n),

        # REMITTANCE HISTORY
        "remit_consistency": np.clip(np.random.beta(4, 3, n), 0, 1),

        # INCOME features
        "base_income": np.random.uniform(700, 1500, n),
        "income_volatility": np.random.uniform(0.15, 0.6, n)
    }

    df = pd.DataFrame(data)

    # Weighted synthetic score (replicates logic in the report, can be different)
    weights = {
        "call_frequency": 0.15,
        "call_duration": 0.05,
        "msg_count": 0.15,
        "commute_consistency": 0.35,
        "night_out_ratio": -0.10,
        "app_learning_news": 0.08,
        "app_betting": -0.30,
        "remit_consistency": 0.25
    }

    df["synthetic_score"] = (
        df[list(weights.keys())] * np.array(list(weights.values()))
    ).sum(axis=1)

    # Normalize scores to a 0–100 range
    df["synthetic_score"] = np.interp(
        df["synthetic_score"],
        (df["synthetic_score"].min(), df["synthetic_score"].max()),
        (0, 100)
    )

    # Assign merit classes (1–5)
    bins = [0, 55, 65, 75, 85, 100]
    labels = [1, 2, 3, 4, 5]
    df["merit_class"] = pd.cut(
        df["synthetic_score"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Reset categories to avoid re-adding the same value if script runs multiple times
    if hasattr(df["merit_class"].dtype, "categories"):
        df["merit_class"] = df["merit_class"].astype("category")
        df["merit_class"] = df["merit_class"].cat.remove_unused_categories()

    # Fill missing values safely
    df["merit_class"] = df["merit_class"].astype(str).replace("nan", "1").astype(int)

    return df


if __name__ == "__main__":
    df_wetrust = generate_wetrust_dataset(100000)
    print("=== SAMPLE OF GENERATED DATA ===")
    print(df_wetrust.head(), "\n")

    df_wetrust.to_csv("wetrust_synthetic_dataset.csv", index=False)
    print("✅ Dataset saved as 'wetrust_synthetic_dataset.csv'")
