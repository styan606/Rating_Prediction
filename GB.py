import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# Load data
print("Loading data...")
df = pd.read_excel(r"C:\Users\styli\OneDrive\Desktop\excel\afterSCALING.xlsx", engine='openpyxl')

# Separate features and target
X = df.drop(columns=['rating'])
y = df['rating']

# Split data into train (60%), validation (20%), and test (20%) while maintaining class distribution
print("Splitting data into train, validation, and test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp)

# Verify class distribution
print("Class distribution in train, validation, and test sets:")
print("Train:\n", y_train.value_counts(normalize=True))
print("Validation:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))

# Train the final model using the best parameters on the combined train+validation set
print("Training final model with best parameters...")
gbc_best = GradientBoostingClassifier(learning_rate=0.2, n_estimators=700)
gbc_best.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

# Final evaluation on the test set
print("Evaluating on the test set...")
y_pred_test = gbc_best.predict(X_test)

print("\nGRADIENT BOOSTING CLASSIFICATION REPORT (Test Set)")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues',
            xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5], annot_kws={"size": 12})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Gradient Boosting (Test Set)')
plt.show()

print("Calculating feature importance using Gradient Boosting built-in method...")
gbc_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': gbc_best.feature_importances_})
gbc_importance_df = gbc_importance_df.sort_values(by='Importance', ascending=False)

print("\nGRADIENT BOOSTING FEATURE IMPORTANCE")
print(gbc_importance_df)

# Plot feature importance
plt.figure(figsize=(14, 6))
sns.barplot(x=gbc_importance_df['Importance'], y=gbc_importance_df['Feature'])
plt.xlabel('Importance')
plt.title('Feature Importance - Gradient Boosting')
plt.show()