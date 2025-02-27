import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from scipy.stats import norm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_excel(r"C:\Users\styli\OneDrive\Desktop\excel\afterSCALING.xlsx", engine='openpyxl')
df = df.sample(frac=1).reset_index(drop=True)

features = df.drop("rating", axis=1)
target = df["rating"]

# Stratified Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_val_idx, test_idx in sss.split(features, target):
    X_train_val, X_test = features.iloc[train_val_idx], features.iloc[test_idx]
    y_train_val, y_test = target.iloc[train_val_idx], target.iloc[test_idx]

sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_idx, val_idx in sss_val.split(X_train_val, y_train_val):
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

# Neural Network class
class PresentationRatingNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout_rate, activation, output_dim):
        super(PresentationRatingNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)


        self.layers.append(nn.Linear(input_dim, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.out(x)

    # Weight Initialization Function
def initialize_weights(model, init_type="normal"):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(layer.weight)
            elif init_type == "xavier":
                nn.init.xavier_normal_(layer.weight)
            elif init_type == "normal":
                nn.init.normal_(layer.weight, mean=0, std=0.2)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # Training function with early stopping
def train_model(model, X_train, y_train, X_val, y_val, batch_size, learning_rate, patience=20, max_epochs=1000, init_type="normal"):
    initialize_weights(model, init_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values - 1, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values - 1, dtype=torch.long)

    best_val_loss = float("inf")
    patience_counter = 0
    best_val_acc = 0

    for epoch in range(max_epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i + batch_size]
            y_batch = y_train_tensor[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            val_loss = criterion(outputs, y_val_tensor)
            _, predictions = torch.max(outputs, 1)
            val_acc = accuracy_score(y_val_tensor.numpy(), predictions.numpy())

        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_acc

# Optuna optimization
def objective(trial):
    batch_size = trial.suggest_int("batch_size", 32, 512)
    hidden_size = trial.suggest_int("hidden_size", 32, 512)
    num_layers = trial.suggest_int("num_layers", 1, 7)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, log=True)
    activation_name = trial.suggest_categorical("activation", ["relu", "leakyrelu", "tanh"])

    activation_functions = {"relu": nn.ReLU(), "leakyrelu": nn.LeakyReLU(), "tanh": nn.Tanh()}
    activation = activation_functions[activation_name]

    model = PresentationRatingNN(X_train.shape[1], hidden_size, num_layers, dropout_rate, activation, len(target.unique()))
    return train_model(model, X_train, y_train, X_val, y_val, batch_size, learning_rate)


# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500)
print("Best hyperparameters found:")
print(study.best_params)

# Calculate required test runs
def calculate_required_test_runs(expected_accuracy, margin_of_error, confidence_level):
    Z = norm.ppf(1 - (1 - confidence_level) / 2)
    return int(np.ceil((Z ** 2 * expected_accuracy * (1 - expected_accuracy)) / (margin_of_error ** 2)))


best_val_acc = study.best_value
required_runs = calculate_required_test_runs(best_val_acc, 0.05, 0.95)
print(f"Required Test Runs: {required_runs}")


best_params = study.best_params
final_model = PresentationRatingNN(
    input_dim=X_train.shape[1],
    hidden_size=best_params["hidden_size"],
    num_layers=best_params["num_layers"],
    dropout_rate=best_params["dropout_rate"],
    activation={"relu": nn.ReLU(), "leakyrelu": nn.LeakyReLU(), "tanh": nn.Tanh()}[best_params["activation"]],
    output_dim=len(target.unique())
)

# Evaluate on the test set after training with early stopping
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values - 1, dtype=torch.long)

# Initialize lists to store results
conf_matrices = []
class_reports = []
test_accuracies = []

# Lists to store results
test_accuracies = []
test_losses = []
conf_matrices = []
class_reports = []

# Define loss function (CrossEntropyLoss for classification)
criterion = torch.nn.CrossEntropyLoss()

# Loop through each required test run
for _ in range(required_runs):
    # Retrain the model
    train_model(final_model, X_train, y_train, X_val, y_val, best_params["batch_size"], best_params["learning_rate"])

    # Evaluate the model on the test set
    final_model.eval()
    with torch.no_grad():
        outputs = final_model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)

        # Compute test loss
        loss = criterion(outputs, y_test_tensor)
        test_losses.append(loss.item())

        # Store accuracy for averaging
        accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
        test_accuracies.append(accuracy)
        print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss.item():.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test_tensor.numpy(), predictions.numpy())
        conf_matrices.append(cm)

        # Classification Report
        report = classification_report(y_test_tensor.numpy(), predictions.numpy(), output_dict=True)
        class_reports.append(report)

# Compute Mean and Standard Deviation
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)

mean_loss = np.mean(test_losses)
std_loss = np.std(test_losses)

# Average Confusion Matrix
avg_cm = np.mean(conf_matrices, axis=0)
avg_cm = np.round(avg_cm).astype(int)

# Average Classification Report (for each class)
avg_class_report = {}

for key in class_reports[0]:
    if isinstance(class_reports[0][key], dict):  # Check if key contains a dict (to exclude 'accuracy' float)
        avg_class_report[key] = {
            'precision': np.mean([class_reports[i][key]['precision'] for i in range(len(class_reports))]),
            'recall': np.mean([class_reports[i][key]['recall'] for i in range(len(class_reports))]),
            'f1-score': np.mean([class_reports[i][key]['f1-score'] for i in range(len(class_reports))]),
        }
    else:  # Handle 'accuracy' separately
        avg_class_report[key] = np.mean([class_reports[i][key] for i in range(len(class_reports))])

# Print average classification report
print("\nAverage Classification Report:")
for class_label, metrics in avg_class_report.items():
    if isinstance(metrics, dict):  # Skip printing accuracy as a dict
        print(f"Class {class_label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1-score']:.4f}")
    else:  # Print accuracy separately
        print(f"Overall Accuracy: {metrics:.4f}")


# Print final results
print(f"\nMean Test Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Mean Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")

# Plot the average confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(avg_cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(1, 6), yticklabels=np.arange(1, 6))
plt.title("Average Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# Plot the average loss and accuracy curves
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, required_runs + 1), test_losses, label='Loss')
plt.xlabel('Test Run')
plt.ylabel('Loss')
plt.title('Average Test Loss Across Runs')

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, required_runs + 1), test_accuracies, label='Accuracy')
plt.xlabel('Test Run')
plt.ylabel('Accuracy')
plt.title('Average Test Accuracy Across Runs')

plt.tight_layout()
plt.show()