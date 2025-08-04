import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Setting seed for reproducibility
keras.utils.set_random_seed(42)

# Load dataset
training_images_filepath = "/content/drive/MyDrive/GTZAN/images_original"
category_labels = os.listdir(training_images_filepath)
xdim, ydim = 180, 180

spectograms = image_dataset_from_directory(
    training_images_filepath, image_size=(xdim, ydim), batch_size=64)

num_batches = tf.data.experimental.cardinality(spectograms).numpy()
train = spectograms.take(num_batches - 2).cache()
remaining = spectograms.skip(num_batches - 2)
validation = remaining.take(1).cache()
test = remaining.skip(1).cache()

# Load VGG16 base model
conv_base = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(xdim, ydim, 3))
conv_base.trainable = False  # Freeze initial layers

# Modified model architecture
inputs = keras.Input(shape=(xdim, ydim, 3))
x = keras.applications.vgg16.preprocess_input(inputs)
x = conv_base(inputs)
x = layers.Conv2D(512, (3,3), activation="relu", padding="same")(x) #new
x = layers.BatchNormalization()(x)  #new
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)  #new
x = layers.BatchNormalization()(x)  #new
x = layers.Dropout(0.5)(x)  #new
x = layers.Dense(256, activation="relu")(x) #new
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)  #new
outputs = layers.Dense(len(category_labels), activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True) #new

# Compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
    metrics=["accuracy"])

# Train model
history = model.fit(
    train, epochs=10, validation_data=validation, callbacks=[early_stopping], verbose=1)

# Fine-tune last few layers
conv_base.trainable = True
for layer in conv_base.layers[:-8]:  # Unfreeze last 8 layers
    layer.trainable = False

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5),
    metrics=["accuracy"])

history = model.fit(
    train, epochs=20, validation_data=validation, callbacks=[early_stopping], verbose=1)

# Evaluate on test set
print("\nEvaluating model on test set...")
predictions_prob = model.predict(test)
predictions = np.argmax(predictions_prob, axis=1)
ground_truth = [label for _, label in test.unbatch()]
ground_truth = tf.stack(ground_truth, axis=0).numpy()

# Calculate and display accuracy
accuracy = accuracy_score(ground_truth, predictions)
print("\nFinal Model Accuracy:", accuracy)

# Generate and display classification report
print("\nClassification Report:")
print(classification_report(ground_truth, predictions, target_names=category_labels))

# Generate and plot confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(ground_truth, predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=category_labels,
            yticklabels=category_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
