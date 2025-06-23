from flask import Flask, request, jsonify, Response,send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import pickle
from keras_facenet import FaceNet
import pyttsx3
import threading
import time

# Flask setup
app = Flask(__name__)
CORS(app)

# FaceNet setup
embedder = FaceNet()

# Ensure directories exist
if not os.path.exists("face_db"):
    os.makedirs("face_db")

# Load face database into memory
def load_face_database():
    db = {}
    for file in os.listdir("face_db"):
        if file.endswith(".pkl"):
            name = file.split(".")[0]
            with open(os.path.join("face_db", file), "rb") as f:
                db[name] = pickle.load(f)
    return db

face_db = load_face_database()
spoken_names = set()
last_reset_time = time.time()

# Tamil speech output using pyttsx3
def speak_tamil_offline(text):
    def speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        voices = engine.getProperty('voices')
        # Attempt to find Tamil-compatible voice (requires pre-installation)
        for voice in voices:
            if 'tamil' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=speak, daemon=True).start()

@app.route("/")
def home():
    return send_file("index.html")


# Register face
@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    image = request.files.get("image")

    if not name or not image:
        return jsonify({"message": "Name or image missing!"}), 400

    img_array = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = embedder.extract(rgb, threshold=0.95)
    if not faces:
        return jsonify({"message": "No face detected!"}), 400

    embedding = faces[0]["embedding"]

    with open(f"face_db/{name}.pkl", "wb") as f:
        pickle.dump(embedding, f)

    face_db[name] = embedding
    return jsonify({"message": f"Face registered for {name}."})

# Remove face
@app.route("/remove", methods=["POST"])
def remove():
    name = request.form.get("name")

    if not name:
        return jsonify({"message": "Name is required!"}), 400

    file_path = f"face_db/{name}.pkl"
    if not os.path.exists(file_path):
        return jsonify({"message": f"Face for {name} not found!"}), 404

    try:
        os.remove(file_path)
        face_db.pop(name, None)
        return jsonify({"message": f"Face for {name} removed successfully."})
    except Exception as e:
        return jsonify({"message": f"An error occurred while removing: {str(e)}"}), 500

# Live recognition
def generate_frames():
    global spoken_names, last_reset_time
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = embedder.extract(rgb, threshold=0.95)

        for face in faces:
            box = face["box"]
            embedding = face["embedding"]
            name = "Unknown"
            color = (0, 0, 255)

            for db_name, db_emb in face_db.items():
                dist = np.linalg.norm(embedding - db_emb)
                if dist < 0.7:
                    name = db_name
                    color = (0, 255, 0)
                    if name not in spoken_names:
                        speak_tamil_offline(name)
                        spoken_names.add(name)
                    break

            left, top, w, h = box
            right, bottom = left + w, top + h
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Reset spoken names every 2 seconds
        if time.time() - last_reset_time > 2:
            spoken_names.clear()
            last_reset_time = time.time()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/recognize")
def recognize():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/list", methods=["GET"])
def list_faces():
    names = [filename.split(".")[0] for filename in os.listdir("face_db") if filename.endswith(".pkl")]
    return jsonify({"names": names})


if __name__ == "__main__":
    app.run(debug=True)
