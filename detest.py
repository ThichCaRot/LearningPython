import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd
import pathlib
import cv2
import time
from firebase import firebase

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180

gender = tf.keras.utils.image_dataset_from_directory(
    "D:/Projects/LearningPython/cleanup",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "D:/Projects/LearningPython/cleanup",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = gender.class_names
print(class_names)

dataset_unbatched = tuple(gender.unbatch())
labels = []
for (image, label) in dataset_unbatched:
    labels.append(label.numpy())
labels = pd.Series(labels)

count = labels.value_counts().sort_index()
count.index = gender.class_names
print(count)

plt.figure(figsize=(10, 10))
for images, labels in gender.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in gender:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

gender = gender.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_ds = gender.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 10
history = model.fit(
    gender,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in gender.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

img_height = 180
img_width = 180

# Firebase Initialization
firebase = firebase.FirebaseApplication('https://sample-fd33f-default-rtdb.firebaseio.com/', None)

def process_webcam_frame(model):
    video_capture = cv2.VideoCapture(0)  # Initialize webcam capture

    while True:
        ret, frame = video_capture.read()  # Read frame from webcam

        # Preprocess the frame
        frame = cv2.resize(frame, (img_width, img_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0

        # Make predictions using the model
        predictions = model.predict(frame)
        score = tf.nn.softmax(predictions[0])
        class_index = np.argmax(score)
        class_name = class_names[class_index]
        confidence = np.max(score) * 100

        # Push data to Firebase database
        data = {'class_name': class_name, 'confidence': confidence}
        result = firebase.post('/predictions', data)  # Push data to '/predictions' endpoint

        # Display the frame with prediction information
        cv2.putText(frame, f'{class_name} ({confidence:.2f}%)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(10)  # Wait for 10 seconds before processing the next frame

    video_capture.release()
    cv2.destroyAllWindows()

# Train the model
epochs = 10
history = model.fit(
    gender,
    validation_data=val_ds,
    epochs=epochs
)

# ... Plotting code ...

# Add the webcam functionality
process_webcam_frame(model)