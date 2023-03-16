import cv2
import numpy as np
import os

# Get full file paths
weights_path = os.path.abspath(r"C:\Users\vaish\Downloads\vdetect\yolov3.weights")
config_path = os.path.abspath(r"C:\Users\vaish\Downloads\vdetect\yolov3.cfg")

# Load YOLOv3 model and configuration
net = cv2.dnn.readNet(weights_path, config_path)



# Define classes
classes = []
with open(r"C:\Users\vaish\Downloads\vdetect\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set input video file
cap = cv2.VideoCapture(r"C:\Users\vaish\Downloads\vdetect\input.mp4")

# Set output video codec and size
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(r"C:\Users\vaish\Downloads\vdetect\output.mp4", fourcc,fps, (width, height))

# Set input image size
input_size = (416, 416)

while True:
    # Read frame from input video
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for YOLOv3 input
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, (0,0,0), swapRB=True, crop=False)

    # Set input and output nodes for YOLOv3 network
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()

    # Forward pass through YOLOv3 network
    outputs = net.forward(output_layers)

    # Find vehicle detections and classes
    vehicles = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2: # Class ID for vehicles is 2
                # Calculate bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                vehicles.append((left, top, width, height))

    # Draw bounding boxes and class labels on frame
    for left, top, width, height in vehicles:
        cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle", (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write output frame to output video
    out.write(frame)

    # Display output frame
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release input and output video
cap.release()
out.release()
cv2.destroyAllWindows()
