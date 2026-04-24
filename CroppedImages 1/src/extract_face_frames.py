import cv2
import os

cascade_path = r"C:\Users\jackm\Downloads\CroppedImages\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

output_dir = r"C:\Users\jackm\Downloads\CroppedImages\src\static\face_frames"
os.makedirs(output_dir, exist_ok=True)

def extract_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return 0, []

    saved_count = 0
    saved_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]

            filename = f"frame_{saved_count:05d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, face_crop)

            web_path = "/static/face_frames/" + filename
            saved_paths.append(web_path)

            saved_count += 1

    cap.release()
    return saved_count, saved_paths
