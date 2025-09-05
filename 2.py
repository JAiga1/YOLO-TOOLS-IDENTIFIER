from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("22.pt")

# Run inference on 'bus.jpg'
results = model(["1.jpg", "2.jpg","3.jpg","4.jpg"])  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()