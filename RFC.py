import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_excel(r"C:\Users\styli\OneDrive\Desktop\excel\afterSCALING.xlsx", engine='openpyxl')

# Separate features and target
X = df.drop(columns=['rating'])
y = df['rating']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)
best_rf_params = grid_search.best_params_
print(best_rf_params)

# Monte Carlo Simulation for average confusion matrix and classification report
n_iterations = 30
conf_matrices = []
reports = []

for i in range(n_iterations):
    rf_best = RandomForestClassifier(**best_rf_params, random_state=np.random.randint(0, 10000))
    rf_best.fit(X_train, y_train)
    y_pred_test = rf_best.predict(X_test)
    conf_matrices.append(confusion_matrix(y_test, y_pred_test, labels=[1, 2, 3, 4, 5]))
    reports.append(classification_report(y_test, y_pred_test, output_dict=True))

# Average Confusion Matrix
avg_conf_matrix = np.mean(conf_matrices, axis=0)

# Plot Average Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(avg_conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.title('Average Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Aggregate Classification Report
from collections import defaultdict

metrics_agg = defaultdict(list)

for report in reports:
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                metrics_agg[f"{label}_{metric_name}"].append(value)
        else:
            metrics_agg[label].append(metrics)

avg_report = {metric_name: np.mean(values) for metric_name, values in metrics_agg.items()}

print("\nAVERAGE CLASSIFICATION REPORT")
for metric_name, mean in avg_report.items():
    print(f"{metric_name}: {mean:.3f}")

# Feature Importance
rf_final = RandomForestClassifier(**best_rf_params, random_state=42)
rf_final.fit(X_train, y_train)
rf_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf_final.feature_importances_})
rf_importance_df = rf_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(14, 6))
sns.barplot(x=rf_importance_df['Importance'], y=rf_importance_df['Feature'])
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.show()
