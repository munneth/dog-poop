from dlclive import DLCLive, Processor
import cv2
import time
from collections import deque

dlc_proc = Processor()
# Use the exported model folder
model_path = r"C:\Users\munne\Documents\DLC\poop_project-munneth-2025-06-25\exported-models\DLC_poop_project_resnet_50_iteration-0_shuffle-1"

# Initialize DLC-Live
dlc_live = DLCLive(model_path)

# Open webcam (or your video capture device)
cap = cv2.VideoCapture(0)

#pose buffer
pose_buffer = deque(maxlen=150)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #get inference
    dlc_live.init_inference(frame)
    
    # Get pose estimation
    pose = dlc_live.get_pose(frame)

    # Optionally overlay the pose
    dlc_live.plot_poses(frame)

    # Display the frame
    cv2.imshow("DLC Live", frame)

    #get y values of tails
    tail1_y = pose[3][1]
    tail2_y = pose[4][1]
    tail3_y = pose[5][1]

    #get y values of spine
    spine1_y = pose[0][1]
    spine2_y = pose[1][1]
    spine3_y = pose[2][1]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

