



from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_models(X_train, X_test, y_train, y_test):
    print("\n--- Model Training Started ---")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n► {name} being trained...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        }

        print(f"{name} Results:")
        print(f" - MAE: {mae}")
        print(f" - MSE: {mse}")
        print(f" - R2:  {r2}")

    print("\n--- Model Training is Over ---")
    return results