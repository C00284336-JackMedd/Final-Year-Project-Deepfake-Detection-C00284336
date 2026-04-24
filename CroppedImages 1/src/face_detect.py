import cv2

cascade_path = r"C:\Users\jackm\Downloads\CroppedImages\haarcascade_frontalface_alt.xml"
image_path = r"C:\Users\jackm\Downloads\CroppedImages\raw_images\sample.jpg"

haar = cv2.CascadeClassifier(cascade_path)
img = cv2.imread(image_path)

faces = haar.detectMultiScale(img, 1.3, 5)
print("Faces detected:", len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
