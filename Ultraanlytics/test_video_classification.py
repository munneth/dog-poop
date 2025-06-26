import cv2
import time
import datetime
import os
from ultralytics import YOLO

# Load your trained classification model
model = YOLO("runs/classify/train4/weights/best.pt")
POSITIVE_LABEL = "poop"

# Output directory for detected clips
output_path = "poop-clips"
os.makedirs(output_path, exist_ok=True)

# Video file to test (change this to your video file path)
video_file = "Security Cam - Dog Poop.mp4"  # Change this to your video file

# Video capture from file
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_file}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

# Parameters
frame_rate = 10
start_thresh = 0.85
stop_thresh = 0.70
recording = False
last_label = ""
video_writer = None
last_write_time = time.time()
frame_count = 0

# For display
cv2.namedWindow('Poop Detection Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Poop Detection Test', 800, 600)

def process_video():
    global recording, last_label, video_writer, frame_count

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Resize frame for classification
        resized = cv2.resize(frame, (224, 224))
        results = model.predict(resized, imgsz=224, save=False, verbose=False)
        probs = results[0].probs
        top_id = probs.top1
        label = model.names[top_id]
        confidence = float(probs.top1conf)

        # Add overlay text
        overlay_text = f"{label} ({confidence:.1%})"
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Recording logic
        if label == "poop":
            if confidence >= start_thresh and not recording:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                out_path = os.path.join(output_path, f"poop_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(out_path, fourcc, frame_rate, (width, height))
                print(f"💩 Pooping started at frame {frame_count}...")
                recording = True
            elif confidence <= stop_thresh and recording:
                print(f"✅ Pooping ended at frame {frame_count}.")
                recording = False
                video_writer.release()
        else:
            if recording and confidence <= stop_thresh:
                print(f"✅ Pooping ended at frame {frame_count}.")
                recording = False
                video_writer.release()

        if recording:
            video_writer.write(frame)
            # Add recording indicator
            cv2.circle(frame, (width - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 80, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Poop Detection Test', frame)
        
        # Press 'q' to quit, 'p' to pause
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)  # Wait for any key to continue

        # Optional: slow down playback for better observation
        time.sleep(1 / frame_rate)

    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🐾 Poop detection test starting...")
    print(f"Testing video: {video_file}")
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print(f"Output will be saved to: {output_path}")
    
    process_video()
    
    print("Test completed!") 