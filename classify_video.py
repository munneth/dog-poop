import cv2
import time
import datetime
import os
from ultralytics import YOLO
from flask import Flask, Response
from threading import Thread

# Load your trained classification model
model = YOLO("best.pt")
POSITIVE_LABEL = "poop"

# Set up Flask app
app = Flask(__name__)

# USB video writer setup
usb_path = "/mnt/poopusb/poop-clips"
os.makedirs(usb_path, exist_ok=True)

# Video capture from default camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Parameters
frame_rate = 10
start_thresh = 0.85
stop_thresh = 0.70
recording = False
last_label = ""
video_writer = None
last_write_time = time.time()

# For MJPEG stream
latest_frame = None

def capture_and_classify():
    global recording, last_label, video_writer, latest_frame, last_write_time

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        resized = cv2.resize(frame, (224, 224))
        results = model.predict(resized, imgsz=224, save=False, verbose=False)
        probs = results[0].probs
        top_id = probs.top1
        label = model.names[top_id]
        confidence = float(probs.top1conf)

        overlay_text = f"{label} ({confidence:.1%})"
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if label == "poop":
            if confidence >= start_thresh and not recording:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                out_path = os.path.join(usb_path, f"poop_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(out_path, fourcc, frame_rate, (640, 480))
                print("💩 Pooping started...")
                recording = True
            elif confidence <= stop_thresh and recording:
                print("✅ Pooping ended.")
                recording = False
                video_writer.release()
        else:
            if recording and confidence <= stop_thresh:
                print("✅ Pooping ended.")
                recording = False
                video_writer.release()

        if recording:
            video_writer.write(frame)

        latest_frame = frame
        time.sleep(1 / frame_rate)

@app.route('/stream.mjpg')
def stream():
    def generate():
        global latest_frame
        while True:
            if latest_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🐾 Poop monitor running – visit http://<pi-ip>:8000/stream.mjpg")
    Thread(target=capture_and_classify, daemon=True).start()
    app.run(host="0.0.0.0", port=8000)
