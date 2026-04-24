import os
from flask import Flask, render_template, request
from predict import predict_image
from extract_face_frames import extract_faces_from_video

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        label, confidence = predict_image(filepath, return_values=True)
        image_path = "/" + filepath.replace("\\", "/")

        return render_template("result.html", image_path=image_path, label=label, confidence=confidence)

    return render_template("index.html")


@app.route("/video", methods=["GET", "POST"])
def video():
    if request.method == "POST":
        file = request.files["video"]
        if file.filename == "":
            return "No video selected", 400

        video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(video_path)

        count, frames = extract_faces_from_video(video_path)

        return render_template("video_result.html", count=count, frames=frames)

    return render_template("video_upload.html")


if __name__ == "__main__":
    app.run(debug=True)
