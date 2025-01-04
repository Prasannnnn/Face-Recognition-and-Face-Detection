import face_recognition
import cv2

# Load the image
image_path = r"C:\Users\ADMIN\Desktop\AIML-Projects\Face-Detection-and-Face-recognition\images\virat_kholi.png"
image = face_recognition.load_image_file(image_path)
landmarks_list = face_recognition.face_landmarks(image)

# Convert to OpenCV BGR format
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Draw landmarks
for landmarks in landmarks_list:
    for feature, points in landmarks.items():
        for point in points:
            cv2.circle(image_bgr, point, 2, (255, 0, 0), -1)  # Blue circles for landmarks

# Show the image
cv2.imshow("Face Landmarks", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
