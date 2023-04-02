<<<<<<< HEAD
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import matplotlib.pyplot as plt

WIDTH, HEIGHT = (1024, 768)

image = tf.io.read_file('D:/Downloads/bump.jpg')
image = tf.io.decode_image(image)
image = tf.image.resize(image, (HEIGHT, WIDTH))
images = tf.expand_dims(image, axis=0) / 255

model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=80,
    training=False,
    yolo_max_boxes=50,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
)

model.load_weights('yolov4.h5')

boxes, scores, classes, detections = model.predict(images)

boxes = boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
scores = scores[0]
classes = classes[0].astype(int)
detections = detections[0]

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop',  'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

plt.imshow(images[0])
ax = plt.gca()

for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):
    if score > 0:
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='green')
        ax.add_patch(rect)

        text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
        ax.text(xmin, ymin, text, fontsize=9, bbox=dict(facecolor='yellow', alpha=0.6))

plt.title('Objects detected: {}'.format(detections))
plt.axis('off')
=======
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import matplotlib.pyplot as plt

WIDTH, HEIGHT = (1024, 768)

image = tf.io.read_file('D:/Downloads/bump.jpg')
image = tf.io.decode_image(image)
image = tf.image.resize(image, (HEIGHT, WIDTH))
images = tf.expand_dims(image, axis=0) / 255

model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=80,
    training=False,
    yolo_max_boxes=50,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
)

model.load_weights('yolov4.h5')

boxes, scores, classes, detections = model.predict(images)

boxes = boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
scores = scores[0]
classes = classes[0].astype(int)
detections = detections[0]

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop',  'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

plt.imshow(images[0])
ax = plt.gca()

for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):
    if score > 0:
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='green')
        ax.add_patch(rect)

        text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
        ax.text(xmin, ymin, text, fontsize=9, bbox=dict(facecolor='yellow', alpha=0.6))

plt.title('Objects detected: {}'.format(detections))
plt.axis('off')
>>>>>>> d71da0d7ab563256195a951ccd3a9f49ce5c3cc3
plt.show()