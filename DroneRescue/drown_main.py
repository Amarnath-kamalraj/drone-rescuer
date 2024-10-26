from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt

# Initialize the Inference HTTP Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="hL2SDabLDZarxOpVSltI"
)

# Start video capture (0 for default camera, or replace with video file path)
cap = cv2.VideoCapture(0)  # Change '0' to a video file path if needed

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is captured

    # Save the current frame temporarily for inference
    img_path = "temp_frame.jpg"
    cv2.imwrite(img_path, frame)

    # Perform inference on the current frame
    result = CLIENT.infer(img_path, model_id="swimmingxdrowning/4")

    # Get original dimensions of the frame
    original_height, original_width = frame.shape[:2]

    # Get dimensions from the inference result
    target_width = result['image']['width']
    target_height = result['image']['height']

    # Resize the image to the specified dimensions
    frame_resized = cv2.resize(frame, (target_width, target_height))

    # Calculate scaling factors for x and y coordinates
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # Check if predictions exist
    if len(result['predictions']) > 0:
        # Extract the bounding box values from the first prediction
        x = result['predictions'][0]['x'] / 2
        y = result['predictions'][0]['y'] / 2
        box_width = result['predictions'][0]['width']
        box_height = result['predictions'][0]['height']

        # Get the class label from the first prediction
        class_label = result['predictions'][0]['class']  # Assuming 'class' contains the label

        # Scale the bounding box values to match the resized image
        x = x * scale_x
        y = y * scale_y
        box_width = box_width * scale_x
        box_height = box_height * scale_y

        # Draw the bounding box on the frame
        start_point = (int(x), int(y))
        end_point = (int(x + box_width), int(y + box_height))
        color = (0, 255, 0)  # Green color for the bounding box
        thickness = 2  # Thickness of the bounding box

        # Draw the bounding box on the original frame
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        # Define the position for the text (slightly above the bounding box)
        text_position = (int(x), int(y) - 10)  # Position the text above the box

        # Add text label to the bounding box
        cv2.putText(frame, class_label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with the bounding box
    cv2.imshow('Real-Time Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
