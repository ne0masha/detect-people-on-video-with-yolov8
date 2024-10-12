from ultralytics import YOLO
import cv2

# use YOLOv8
model = YOLO('yolov8n.pt')


# load video
video_path = 'input/input_video.mp4'
cap = cv2.VideoCapture(video_path)

# get the output_video parameters
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# specify the output_video parameters
output_path = 'output/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

ret = True

# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # detect objects
        results = model.track(frame, persist=True)

        # write the result in output_video
        out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# freeing up resources
cap.release()
cv2.destroyAllWindows()
