import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def handle_missing_values(df):
    return df.replace({np.nan: 0})


def remove_multicollinearity(df, threshold=0.8):
    corr_matrix = df.corr(numeric_only=True)
    to_drop = set()
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                to_drop.add(cols[j])

    return df.drop(columns=list(to_drop), errors="ignore")


def split_features_target(df, target="rainfall"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_val_test_split(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler
