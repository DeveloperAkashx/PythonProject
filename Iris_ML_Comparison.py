import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

print("Iris Classification Project - Pandas + NumPy + Scikit-learn + TensorFlow\n")
print("Loading data...\n")

# Pandas & NumPy Exploration 
iris = load_iris()
X = iris.data
y = iris.target

# Pandas DataFrame for easy exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("First 8 rows of the dataset:")
print(df.head(8))
print("\nDataset statistics:")
print(df.describe())

# NumPy operations
print("\nNumPy calculations:")
print(f"Mean of each feature   : {np.mean(X, axis=0).round(3)}")
print(f"Std  of each feature   : {np.std(X, axis=0).round(3)}")
print(f"Correlation matrix (NumPy):")
print(np.round(np.corrcoef(X.T), 3))

# Train/Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {X_train_scaled.shape[0]}")
print(f"Test samples    : {X_test_scaled.shape[0]}")

# Scikit-learn: Random Forest
print("\nTraining RandomForest (Scikit-learn)...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"RandomForest Accuracy : {acc_rf:.4f}")
print("Classification Report (RandomForest):")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

# TensorFlow/Keras: Neural Network
print("\nTraining Neural Network (TensorFlow/Keras)...")

# One-hot encode labels for Keras
y_train_cat = to_categorical(y_train, 3)
y_test_cat = to_categorical(y_test, 3)

model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled,
    y_train_cat,
    epochs=80,
    batch_size=16,
    validation_split=0.1,
    verbose=0
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
y_pred_tf = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)

print(f"TensorFlow Neural Net Accuracy: {test_acc:.4f}")

# Final Comparison
print("\n" + "="*50)
print("FINAL RESULTS".center(50))
print("="*50)
print(f"{'Model':<25} {'Accuracy':>10}")
print("-"*50)
print(f"{'RandomForest (sklearn)':<25} {acc_rf:>10.4f}")
print(f"{'Neural Network (TF)':<25} {test_acc:>10.4f}")
print("="*50)

if test_acc > acc_rf:
    print("The TensorFlow neural network won! ")
else:
    print("RandomForest won (or tied)! ")