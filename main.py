from src.data_loader import load_data


df = load_data()


print("\n--- First 5 Lines ---")
print(df.head())

from src.explore import basic_info, plot_happiness_distribution

#show basic information
basic_info(df)

# Happiness distribution graph
plot_happiness_distribution(df)

from src.preprocess import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data(df)

from src.model_training import train_models

results = train_models(X_train, X_test, y_train, y_test)
print("\n--- Model Results ---")
print(results)


from src.model_plots import plot_predictions, plot_error_distribution, plot_feature_importance
from src.model_training import train_models


results = train_models(X_train, X_test, y_train, y_test)


from sklearn.ensemble import RandomForestRegressor

best_model = RandomForestRegressor(n_estimators=200, random_state=42)
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)

# Graphics
plot_predictions(y_test, preds)
plot_error_distribution(y_test, preds)

# Feature Importance
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
plot_feature_importance(best_model, feature_names)


