import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def plot_distribution(df, target):
    sns.countplot(x=df[target])
    plt.show()


def plot_heatmap(df):
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.show()


def compare_models(X_train, y_train):
    models = {
        "lr": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier()
    }

    scores = []

    for model in models.values():
        scores.append(cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean())

    sns.barplot(x=list(models.keys()), y=scores)
    plt.show()


def plot_conf_matrix(model1, model2, X_val, y_val):
    fig, ax = plt.subplots(1, 2)

    sns.heatmap(confusion_matrix(y_val, model1.predict(X_val)), ax=ax[0])
    sns.heatmap(confusion_matrix(y_val, model2.predict(X_val)), ax=ax[1])

    plt.show()
