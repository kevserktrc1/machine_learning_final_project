
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_test, predictions):
    """
    Scatter plot: Actual vs Predicted Happiness Index
    """
    plt.figure()
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel("Actual Happiness Index")
    plt.ylabel("Predicted Happiness Index")
    plt.title("Actual vs Predicted Happiness Index")

    # Reference line (perfect prediction)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.show()


def plot_error_distribution(y_test, predictions):
    """
    Histogram of prediction errors
    """
    errors = y_test - predictions

    plt.figure()
    plt.hist(errors, bins=20)
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Feature importance bar chart (Random Forest)
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure()
    plt.barh(range(len(importances)), importances[indices])
    plt.yticks(range(len(importances)), np.array(feature_names)[indices])
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.show()