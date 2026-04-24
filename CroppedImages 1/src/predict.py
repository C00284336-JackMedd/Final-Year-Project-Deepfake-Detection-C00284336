import sys
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from mesonet import Meso4

# Load model
weights_path = r"C:\Users\jackm\Downloads\CroppedImages\best_weights.h5"
model = Meso4()
model.model.load_weights(weights_path)

# Haar cascade
cascade_path = r"C:\Users\jackm\Downloads\CroppedImages\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_and_crop_face(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not load image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        raise ValueError("No face detected.")

    # Choose the largest face (more reliable)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    cropped = img[y:y+h, x:x+w]
    return cropped

def preprocess(img):
    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 128x128
    img = cv2.resize(img, (128, 128))

    # Normalize exactly like training
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path, return_values=False):
    face = detect_and_crop_face(image_path)
    processed = preprocess(face)

    pred = model.model.predict(processed)[0][0]

    # MesoNet convention:
    # 0 = REAL
    # 1 = FAKE
    label = "FAKE" if pred >= 0.5 else "REAL"
    confidence = pred if pred >= 0.5 else 1 - pred

    if return_values:
        return label, float(f"{confidence:.4f}")

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image>")
        sys.exit(1)

    predict_image(sys.argv[1])
