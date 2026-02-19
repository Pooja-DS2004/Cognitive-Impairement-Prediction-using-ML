# ============================
# Python 3.8 Compatible Script
# Transfer Learning with MobileNetV2 (Local Weights)
# Separate plots for Accuracy, Precision, Recall, F1 score
# ============================

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score

# ============================
# 1. Dataset Setup
# ============================
data_dir = "./Dataset"   # Make sure this directory contains subfolders for each class
img_size = 224
batch_size = 16


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# ============================
# 2. Build Model (Transfer Learning with Local Weights)
# ============================

base_model = MobileNetV2(weights=None, include_top=False, input_shape=(img_size, img_size, 3))
base_model.load_weights("mobilenet_v2_weights.h5")
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", Precision(name='precision'), Recall(name='recall')]
)

model.summary()

# ============================
# 3. Training
# ============================

checkpoint = ModelCheckpoint("bests_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=[checkpoint, early_stop]
)

# ============================
# 4. Calculate F1 Score on Validation Set
# ============================

val_data.reset()
val_preds_prob = model.predict(val_data)
val_preds = (val_preds_prob > 0.5).astype(int).flatten()
val_true = val_data.classes
f1 = f1_score(val_true, val_preds)
print(f"F1 Score on validation set: {f1:.4f}")

# ============================
# 5. Plotting 4 Separate Graphs
# ============================

epochs_range = range(len(history.history['accuracy']))

# 1. Accuracy Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 2. Precision Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history.history['precision'], label='Train Precision')
plt.plot(epochs_range, history.history['val_precision'], label='Val Precision')
plt.title("Precision Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()

# 3. Recall Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history.history['recall'], label='Train Recall')
plt.plot(epochs_range, history.history['val_recall'], label='Val Recall')
plt.title("Recall Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)
plt.show()

# 4. F1 Score (Single value, horizontal line)
plt.figure(figsize=(8, 5))
plt.bar(['F1 Score (Val Set)'], [f1], color='blue')
plt.ylim(0, 1)
plt.title("F1 Score on Validation Set")
plt.ylabel("F1 Score")
plt.grid(axis='y')
plt.show()

# 4b. Confusion Matrix on Validation Set
# ============================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Compute confusion matrix
cm = confusion_matrix(val_true, val_preds)

# Display it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_data.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Validation Set")
plt.grid(False)
plt.show()
