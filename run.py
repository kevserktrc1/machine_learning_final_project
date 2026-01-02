import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.data_loader import load_data
from src.preprocess import preprocess_data

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_experiment():
    print("Running full experiment pipeline...")

    results_dir = os.path.join(os.getcwd(), "results")
    plots_dir = os.path.join(results_dir, "plots")
    ensure_dir(plots_dir)

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    metrics = {}
    pdf_pages = PdfPages(os.path.join(results_dir, "presentation.pdf"))

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        metrics[name] = {"MAE": float(mae), "MSE": float(mse), "R2": float(r2)}

        # Plot: Actual vs Predicted
        plt.figure()
        plt.scatter(y_test, preds, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--", color="k")
        plt.xlabel("Actual Happiness Index")
        plt.ylabel("Predicted Happiness Index")
        plt.title(f"{name} — Actual vs Predicted")
        fig_path = os.path.join(plots_dir, f"{name.replace(' ', '_')}_actual_vs_pred.png")
        plt.savefig(fig_path)
        pdf_pages.savefig()
        plt.close()

        # Plot: Error distribution
        errors = (y_test - preds).ravel()
        plt.figure()
        plt.hist(errors, bins=20)
        plt.title(f"{name} — Prediction Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        fig_path = os.path.join(plots_dir, f"{name.replace(' ', '_')}_error_dist.png")
        plt.savefig(fig_path)
        pdf_pages.savefig()
        plt.close()

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            try:
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                plt.figure(figsize=(8, 6))
                plt.title(f"{name} — Feature Importances")
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), indices, rotation=90)
                fig_path = os.path.join(plots_dir, f"{name.replace(' ', '_')}_feat_importance.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                pdf_pages.savefig()
                plt.close()
            except Exception:
                pass

        # Learning curve (R2)
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train, cv=5, scoring="r2", train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1
            )
            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)

            plt.figure()
            plt.plot(train_sizes, train_mean, label="Train R2")
            plt.plot(train_sizes, test_mean, label="CV R2")
            plt.xlabel("Training set size")
            plt.ylabel("R2 score")
            plt.title(f"{name} — Learning Curve (R2)")
            plt.legend()
            fig_path = os.path.join(plots_dir, f"{name.replace(' ', '_')}_learning_curve.png")
            plt.savefig(fig_path)
            pdf_pages.savefig()
            plt.close()
        except Exception:
            pass

    pdf_pages.close()

    # Save metrics
    metrics_path = os.path.join(results_dir, "model_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("Experiment finished. Results saved to 'results/' and presentation.pdf created.")


if __name__ == "__main__":
    run_experiment()
