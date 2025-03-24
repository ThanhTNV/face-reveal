import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
import tensorflow as tf
from mtcnn import MTCNN
import numpy as np
from google.colab.patches import cv2_imshow

# Load MobileNetV2 as a feature extractor
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
for layer in base_model.layers:
    layer.trainable = False

# Extract features from training images
train_dir = '/content/data'
# Create train_dir if it doesn't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print(f"Created directory: {train_dir}")
  
persons = os.listdir(train_dir)
features = []
labels = []

model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
for person in persons:
    person_dir = os.path.join(train_dir, person)
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = feature_extractor.predict(x)
        features.append(feature[0])
        labels.append(person)

features = np.array(features)
labels = np.array(labels)

# Train k-NN classifier
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
clf = KNeighborsClassifier(n_neighbors=1, metric='cosine')
clf.fit(features, labels_encoded)

# Test with a new image
test_img_path = '/content/bach.jpg'
img = image.load_img(test_img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
test_feature = feature_extractor.predict(x)
predicted_label = clf.predict(test_feature)
person_name = le.inverse_transform(predicted_label)[0]

print(f"The predicted person is: {person_name}")


# Load MobileNetV2 for feature extraction (if needed)
mobilenet = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Initialize the MTCNN face detector
detector = MTCNN()

# Load your image (make sure to change 'path_to_your_image.jpg' to your image file path)
image_path = '/content/bach.jpg'

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
else:
    # Detect faces in the image
    faces = detector.detect_faces(image)

    # Process each detected face
    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)
        width, height = max(0, width), max(0, height)

        # Draw a bounding box around the face
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

    # Display the image with detected faces
    cv2_imshow(image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()