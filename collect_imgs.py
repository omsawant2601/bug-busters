import os
import cv2

# Define dataset directory
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Parameters
number_of_classes = 35  # Number of classes
dataset_size = 25  # Number of images per class

# Try different camera indices if needed (0, 1, 2, etc.)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera. Please check the connection or try a different index.")
    exit()

# Loop through each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Prompt user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Check your camera connection.")
            break  # Exit loop if frame capture fails

        cv2.putText(frame, 'Ready? Press "Q" to start!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Collect dataset images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break  # Stop collecting if the frame is invalid

        cv2.imshow('frame', frame)
        cv2.waitKey(25)  # Small delay to allow frame display

        # Save image
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")

        counter += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()