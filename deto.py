import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd
import pathlib
import cv2
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import paho.mqtt.client as mqtt

batch_size = 32
img_height = 180
img_width = 180

mqtt_broker = 'mqtt.flespi.io'
mqtt_port = 1883
mqtt_username = 'eco7k4WNUKZYNP2SxQmcxDP5SN3n8qBcrP7BHTdrs0d3F3L0JV14pE05fRid8Idp'
data_log = 'prediction/result'


# MQTT callback functions
def on_connect(client, userdata, flags, rc):
    print('Connected to MQTT broker')
    client.subscribe([(data_log, 0)])

client = mqtt.Client()
client.username_pw_set(username=mqtt_username)
client.on_connect = on_connect
# client.on_message = on_message
client.connect(mqtt_broker, mqtt_port, 60)
#  client.loop_forever()
client.loop_start() 

# Load the gender dataset from directory
cleanup = tf.keras.utils.image_dataset_from_directory(
  "D:/Projects/LearningPython/cleanup",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "D:/Projects/LearningPython/cleanup",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = cleanup.class_names
print(class_names)

dataset_unbatched = tuple(cleanup.unbatch())
labels = []
for (image, label) in dataset_unbatched:
    labels.append(label.numpy())
labels = pd.Series(labels)

# Adjustments
count = labels.value_counts().sort_index()
count.index = cleanup.class_names
print(count)

plt.figure(figsize=(10, 10))
for images, labels in cleanup.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in cleanup:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

cleanup = cleanup.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_ds = cleanup.map(lambda x, y: (normalization_layer(x), y))
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

epochs = 1
history = model.fit(
  cleanup,
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
for images, _ in cleanup.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

#Function to capture video from webcam
# def capture_video_from_webcam():
#     cap = cv2.VideoCapture(0)
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         cv2.imshow('Webcam', frame)
#         if frame_count % 150 == 0:
#             predict_frame(frame)
#         frame_count += 1
#         if cv2.waitKey(1) == 27:  # ESC key
#             break
#     cap.release()
#     cv2.destroyAllWindows()

#Function to predict the class of an image
def predict_frame(frame):
    # resized_frame = cv2.resize(frame, (img_height, img_width))
    # normalized_frame = resized_frame / 255.0
    # reshaped_frame = np.expand_dims(normalized_frame, axis=0)
    # predictions = model.predict(reshaped_frame)
    # predicted_class = np.argmax(predictions[0])
    # print("Predicted class:", class_names[predicted_class])
    img = cv2.resize(frame, (img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    result_message = "{}".format(predicted_class)
    print(result_message)

    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )

    topic = "prediction/result"
    client.publish(topic, result_message)

def capture_and_predict_frames(interval):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        if frame_count % interval == 0:
            predict_frame(frame)
        
        frame_count += 1
        if cv2.waitKey(1) == 27:  # ESC key
            break
        
        # Check if 5 seconds have passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            predict_frame(frame)
            start_time = time.time()
    
    cap.release()
    cv2.destroyAllWindows()

capture_and_predict_frames(150)

# # Capture video from webcam and predict classes
# # capture_video_from_webcam()

# img = tf.keras.utils.load_img(
#     "D:/Projects/LearningPython/luth.jpg", target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )