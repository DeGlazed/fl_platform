import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

matrix = pd.read_csv('matrix.csv')
print(matrix.head())
print(matrix.info())

conf_matrix = matrix.values
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

n_classes = conf_matrix.shape[0]
precision = np.zeros(n_classes)
recall = np.zeros(n_classes)
f1_score = np.zeros(n_classes)

for i in range(n_classes):
    tp = conf_matrix[i, i]
    fp = np.sum(conf_matrix[:, i]) - tp
    fn = np.sum(conf_matrix[i, :]) - tp
    
    precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

results = pd.DataFrame({
    'Class': range(n_classes),
    'Precision': precision,
    'Recall': recall,
    'F1_Score': f1_score
})

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Per-class metrics:")
print(results)

#Label mapping
#{'bike': 0, 'bus': 1, 'car': 2, 'subway': 3, 'taxi': 4, 'train': 5, 'walk': 6}