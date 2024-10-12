from ultralytics import YOLO
import cv2

# use YOLOv8
model = YOLO('yolov8n.pt')


# load video
video_path = 'input/input_video.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # detect objects
        results = model.track(frame, persist=True)

        # show the result
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# freeing up resources
cap.release()
cv2.destroyAllWindows()
