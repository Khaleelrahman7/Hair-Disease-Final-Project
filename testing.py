import tensorflow as tf
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the saved model
model = tf.keras.models.load_model('scalp_disease_model.keras')

# Create a dictionary to map class indices to disease descriptions
class_names = ['dandruff Hair', 'Folliculitis', 'Head Lice', 'No Disease', 'Psoriasis', 'Tinea']

# Function to preprocess the image
def preprocess_image(image, target_size=(150, 150)):
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to load dataset images for confirmation
def load_dataset_images(dataset_path, class_names, target_size=(150, 150)):
    images = []
    labels = []
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = preprocess_image(img, target_size)
            images.append(img)
            labels.append(class_index)
    return np.vstack(images), np.array(labels)

# Load dataset images for confirmation
dataset_path = r'C:\Users\Admin\PycharmProjects\scalp_disease_detection\Scalp disease\train'  # Update with the path to your dataset
dataset_images, dataset_labels = load_dataset_images(dataset_path, class_names)

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Make prediction
    prediction = model.predict(processed_frame)
    class_id = np.argmax(prediction, axis=1)[0]

    # Get the predicted class images for confirmation
    predicted_class_images = dataset_images[dataset_labels == class_id]

    # Compute similarity with dataset images
    similarities = []
    for img in predicted_class_images:
        sim = cosine_similarity(processed_frame.flatten().reshape(1, -1), img.flatten().reshape(1, -1))
        similarities.append(sim[0][0])

    # Average similarity score
    avg_similarity = np.mean(similarities)

    # Check similarity threshold
    threshold = 0.7  # Adjust the threshold as needed
    if avg_similarity > threshold:
        description = class_names[class_id]
    else:
        description = 'No Disease'

    # Display the prediction and description
    cv2.putText(frame, f'Class: {class_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Description: {description}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('Scalp Disease Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
