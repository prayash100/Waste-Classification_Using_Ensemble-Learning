import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load predictions for biodegradable images (actual = 0)
biodegradable_preds = np.load("staking/meta_predictions_biodegradable.npy")
biodegradable_actuals = np.zeros_like(biodegradable_preds)  # Biodegradable is class 0

# Load predictions for non-biodegradable images (actual = 1)
non_biodegradable_preds = np.load("staking/meta_predictions_non_biodegradable.npy")
non_biodegradable_actuals = np.ones_like(non_biodegradable_preds)  # Non-biodegradable is class 1

# Combine actual and predicted labels
y_true = np.concatenate([biodegradable_actuals, non_biodegradable_actuals])
y_pred = np.concatenate([biodegradable_preds, non_biodegradable_preds])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Ensure cm is a NumPy array before raveling
cm = np.array(cm)

# Extract values
TN, FP, FN, TP = cm.ravel()

# Print confusion matrix and metrics
print("Confusion Matrix (Absolute Values):")
print(cm)

print("\nMetrics:")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# Calculate accuracy
accuracy = ((TP + TN) / (TP + TN + FP + FN))*100
print(f"\nAccuracy: {accuracy:.2f}%")

# --- Absolute Heatmap ---
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Biodegradable (0)", "Non-Biodegradable (1)"], 
            yticklabels=["Biodegradable (0)", "Non-Biodegradable (1)"])
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix (Absolute)")
plt.show()

# --- Normalized Heatmap (0 to 1 scale) ---
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Row-wise normalization

plt.figure(figsize=(6, 4))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=["Biodegradable (0)", "Non-Biodegradable (1)"], 
            yticklabels=["Biodegradable (0)", "Non-Biodegradable (1)"])
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix (Normalized: 0 to 1 Scale)")
plt.show()
