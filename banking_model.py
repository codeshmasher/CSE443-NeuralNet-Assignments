# banking_model.py
# Problem Set 02 - Banking Dataset Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

# 1. Load dataset (semicolon separator!)
data = pd.read_csv("C:/Users/USER/Downloads/bank-data/bank-data/bank-full.csv", sep=';')

# Convert target column 'y' to binary
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Encode categorical features (exclude target)
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != 'y']
for col in cat_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Separate features and target
X = data.drop(columns=['y'])
y = data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Build model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train model (set epochs = 10)
EPOCHS = 10
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)

# 4. Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Plot training history
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()

plt.tight_layout()
plt.show()
