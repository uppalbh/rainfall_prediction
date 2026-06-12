from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {
        "train_acc": accuracy_score(y_train, model.predict(X_train)),
        "val_acc": accuracy_score(y_val, model.predict(X_val)),
        "test_acc": accuracy_score(y_test, model.predict(X_test)),
        "report": classification_report(y_val, model.predict(X_val)),
        "confusion_matrix": confusion_matrix(y_val, model.predict(X_val))
    }
    return results
