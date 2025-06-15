import cv2
from ultralytics import YOLO

# Load your trained classification model
model = YOLO("runs/classify/train3/weights/best.pt")

# Path to the .mp4 video you want to test
video_path = "sample_clip.mp4"
cap = cv2.VideoCapture(video_path)

# Optionally save output
save_output = True
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match training image size (e.g. 224x224)
    resized = cv2.resize(frame, (224, 224))

    # YOLO classification expects a batch of images or a single image
    results = model.predict(resized, imgsz=224, save=False, verbose=False)

    # Get top prediction label
    label = results[0].names[results[0].probs.top1]
    conf = results[0].probs.data[results[0].probs.top1].item()

    # Annotate the original frame
    text = f"{label} ({conf*100:.1f}%)"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("YOLOv8 Classification", frame)
    if save_output:
        out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()

