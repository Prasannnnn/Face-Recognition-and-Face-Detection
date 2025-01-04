import face_recognition
import cv2
import numpy as np

# Load the image
image = face_recognition.load_image_file(r"C:\Users\ADMIN\Desktop\AIML-Projects\Face-Detection-and-Face-recognition\image.png")
landmarks_list = face_recognition.face_landmarks(image)

# Convert to OpenCV BGR format
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Apply digital makeup
for landmarks in landmarks_list:
    # Add lipstick (red on lips)
    lip_points = landmarks['top_lip'] + landmarks['bottom_lip']
    cv2.fillPoly(image_bgr, [np.array(lip_points, np.int32)], (0, 0, 255))  # Red color
    
    # Add eyeliner (black around eyes)
    left_eye = np.array(landmarks['left_eye'], np.int32)
    right_eye = np.array(landmarks['right_eye'], np.int32)
    cv2.polylines(image_bgr, [left_eye], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(image_bgr, [right_eye], isClosed=True, color=(0, 0, 0), thickness=2)

# Show the image
cv2.imshow("Digital Makeup", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
