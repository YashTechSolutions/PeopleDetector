from ultralytics import YOLO
import cv2
import math

# Initialize the YOLO model with the source path to avoid warnings
model = YOLO("yolo-Weights/yolov10x.pt")

# Object classes
classNames = ["person"]

# Use OpenCV to capture video from the webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Run the model on the frame
    results = model(img, stream=True)

    # Loop through the detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls >= len(classNames):
              
                continue
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence level
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Get the class name for the detected object
            
            print("Class name -->", classNames[cls])

            # Display the object class on the frame
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Display the frame
    cv2.imshow('Webcam', img)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
