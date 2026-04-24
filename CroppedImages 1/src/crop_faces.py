import cv2
import glob
import os

cascade_path = r"C:\Users\jackm\Downloads\CroppedImages\final_cascade3.xml"
input_path = r"C:\Users\jackm\Downloads\CroppedImages\raw_images\*.*"
output_path = r"C:\Users\jackm\Downloads\CroppedImages\cropped"

os.makedirs(output_path, exist_ok=True)

cascade = cv2.CascadeClassifier(cascade_path)
img_number = 1

for file in glob.glob(input_path):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected:", file)
        continue

    x, y, w, h = faces[0]
    roi = img[y:y+h, x:x+w]
    resized = cv2.resize(roi, (128, 128))

    save_path = os.path.join(output_path, f"{img_number}.jpg")
    cv2.imwrite(save_path, resized)
    img_number += 1
