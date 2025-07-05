from dlclive import DLCLive
import cv2

# Use the exported model folder
model_path = r"C:\Users\munne\Documents\DLC\poop_project-munneth-2025-06-25\exported-models\DLC_poop_project_resnet_50_iteration-0_shuffle-1"

# Initialize DLC-Live
dlc_live = DLCLive(model_path)

# Open webcam (or your video capture device)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get pose estimation
    pose = dlc_live.get_pose(frame)

    # Optionally overlay the pose
    dlc_live.plot_poses(frame)

    # Display the frame
    cv2.imshow("DLC Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

