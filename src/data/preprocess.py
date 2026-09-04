import numpy as np
import pandas as pd


def preprocess_netflix_data(df):
    """Preprocess Netflix data"""
    df = df.copy()

    # Handle missing values
    df["director"].fillna("Unknown", inplace=True)
    df["cast"].fillna("Unknown", inplace=True)
    df["country"].fillna("Unknown", inplace=True)
    df["rating"].fillna(df["rating"].mode()[0], inplace=True)
    df["duration"].fillna(df["duration"].mode()[0], inplace=True)

    # Convert date
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    # Create combined features
    df["combined_features"] = df.apply(
        lambda row: " ".join(
            [
                str(row["director"]) if row["director"] != "Unknown" else "",
                str(row["cast"]) if row["cast"] != "Unknown" else "",
                str(row["listed_in"]) if row["listed_in"] != "Unknown" else "",
                str(row["description"]) if row["description"] != "Unknown" else "",
            ]
        ),
        axis=1,
    )

    return df
