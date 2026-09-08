# ============================================================
# Problem One: Convolutional Neural Network (CNN) Assignment
# Author: Mohammad Irfan Ullah
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# ============================================================
# Step 2: Dataset Preparation
# ============================================================

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    "dataset/train",
    target_size=(128,128),
    color_mode="grayscale",
    batch_size=16,
    class_mode="binary"
)

test_data = datagen.flow_from_directory(
    "dataset/test",
    target_size=(128,128),
    color_mode="grayscale",
    batch_size=16,
    class_mode="binary"
)

val_data = datagen.flow_from_directory(
    "dataset/val",
    target_size=(128,128),
    color_mode="grayscale",
    batch_size=16,
    class_mode="binary"
)

print("Class labels:", train_data.class_indices)
print("Training samples:", train_data.samples)
print("Testing samples:", test_data.samples)
print("Validation samples:", val_data.samples)

# ============================================================
# Step 3: Build CNN Model
# ============================================================

model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nModel Summary:")
print(model.summary())

# ============================================================
# Step 4: Train Model (verbose=2 to reduce output)
# ============================================================

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    verbose=2   # one line per epoch
)

# ============================================================
# Step 5: Evaluate Model
# ============================================================

loss, acc = model.evaluate(test_data, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

# ============================================================
# Step 6: Visualize Training Results
# ============================================================

print("History keys:", history.history.keys())

plt.figure(figsize=(8,5))
plt.plot(history.history.get('accuracy', []), label='Training Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
plt.title("CNN Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Optional: Plot Loss curves
plt.figure(figsize=(8,5))
plt.plot(history.history.get('loss', []), label='Training Loss')
plt.plot(history.history.get('val_loss', []), label='Validation Loss')
plt.title("CNN Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
