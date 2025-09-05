import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")  # Replace with your model path

# Load the image
image_path = "1234.jpeg"  # Replace with your image path
frame = cv2.imread(image_path)

# Run YOLOv8 inference on the image
results = model(frame)

# Visualize the results on the image
annotated_frame = results[0].plot()  # Annotate the image with detection results

# Display the annotated image
cv2.imshow("YOLOv8 Inference", annotated_frame)

# Wait for a key press and close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()
