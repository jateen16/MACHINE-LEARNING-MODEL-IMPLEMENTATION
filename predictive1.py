# job_market_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the CSV datacdc
df = pd.read_csv(r"C:\Users\janha\Desktop\jateen\job_market_unemployment_trends.csv")

# 2. Create binary target: 'High' if above median, else 'Low'
median_rate = df['unemployment_rate'].median()
df['unemployment_label'] = df['unemployment_rate'].apply(lambda x: 1 if x > median_rate else 0)

# 3. Select features
features = ['location', 'job_postings', 'average_age', 'college_degree_percentage']
X = df[features]
y = df['unemployment_label']

# 4. One-hot encode location (categorical)
X = pd.get_dummies(X, columns=['location'], drop_first=True)

# 5. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 6. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# 7. Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Keeps the window open when run by double-clicking
input("\nPress Enter to exit...")

import matplotlib.pyplot as plt
import numpy as np

# Count actual and predicted values
actual_counts = np.bincount(y_test)
pred_counts = np.bincount(y_pred)

# Align lengths (in case one class is missing)
if len(pred_counts) < 2:
    pred_counts = np.append(pred_counts, 0)
if len(actual_counts) < 2:
    actual_counts = np.append(actual_counts, 0)

# Bar chart
labels = ['Low', 'High']
x = np.arange(len(labels))  # label locations
width = 0.35  # bar width

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, actual_counts, width, label='Actual')
plt.bar(x + width/2, pred_counts, width, label='Predicted')

plt.ylabel('Number of Samples')
plt.title('Actual vs Predicted Unemployment Class')
plt.xticks(x, labels)
plt.legend()

plt.tight_layout()
plt.show()
