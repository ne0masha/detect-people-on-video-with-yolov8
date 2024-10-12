from ultralytics import YOLO
import cv2
import os

# use YOLOv8
model = YOLO('yolov8n.pt')

video_path = 'input/input_video.mp4'

# check if the input file exists
if not os.path.isfile(video_path):
    print("Error: input video doesn't exist.")
    exit(1)

# load video
cap = cv2.VideoCapture(video_path)

# get the output_video parameters
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# specify the output_video parameters
output_path = 'output/output_video.mp4'
output_dir = 'output'
if not os.path.exists(output_path):
    os.makedirs(output_dir)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

ret = True

# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # detect objects
        results = model.track(frame, persist=True, conf=0.3)

        # result processing: left only people
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls[0].item())]

                if class_name == 'person':
                    bbox = box.xyxy[0].numpy()  # transform to numpy array
                    confidence = box.conf[0].item()

                    # draw a box
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box

        # write the result in output_video
        out.write(frame)

        # to break the processing
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# freeing up resources
cap.release()
cv2.destroyAllWindows()
