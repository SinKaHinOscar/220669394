import cv2
import serial
import time
import random

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up serial communication with Arduino
arduino = serial.Serial('COM3', 9600) # Replace 'COM3' with the port number of your Arduino
time.sleep(2) # Wait for Arduino to initialize

# Set up servos
pan_servo = 90 # Initial position of pan servo
tilt_servo = 90 # Initial position of tilt servo
pan_angle = 180/5 # Number of degrees per step for pan servo
tilt_angle = 180/5 # Number of degrees per step for tilt servo

# Select a random object from the COCO object list
object_name = random.choice(classes)

# Set up video capture device
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Detect objects in the current frame using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == object_name:
                # Object of interest detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Track the object of interest with servos
    if len(boxes) > 0:
        # Get the center of the bounding box of the first detected object
        x, y, w, h = boxes[0]
        object_center_x = x + w/2
        object_center_y = y + h/2

        # Calculate the error between the object center and the center of the frame
        error_x = object_center_x - frame.shape[1]/2
        error_y = object_center_y - frame.shape[0]/2

        # Update the servo positions based on the error
        pan_servo += int(error_x/pan_angle)
        tilt_servo += int(error_y/tilt_angle)

        # Limit the servo positions to within the range of 0 to 180 degrees
        pan_servo = max(min(pan_servo, 180), 0)
        tilt_servo = max(min(tilt_servo, 180), 0)

        # Send the servo positions to the Arduino
        arduino.write(f"{pan_servo},{tilt_servo}\n".encode())

    # Display the resulting image with the object name and confidence
    for i in range(len(boxes)):
        left, top, width, height = boxes[i]
        cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
        cv2.putText(frame, f"{object_name} {confidences[i]:.2f}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
