import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_models(X_train, y_train):
    models = {
        "lr": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(random_state=42)
    }

    trained = {}

    for name, model in models.items():
        cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def save_models(models, path):
    with open(path, "wb") as f:
        pickle.dump(models, f)
