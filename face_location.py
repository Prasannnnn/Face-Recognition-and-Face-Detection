import face_recognition
import cv2

# Load the image
image = face_recognition.load_image_file(r"C:\Users\ADMIN\Desktop\AIML-Projects\Face-Detection-and-Face-recognition\image.png")
face_locations = face_recognition.face_locations(image)

# Convert to OpenCV BGR format
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Draw rectangles around faces
for top, right, bottom, left in face_locations:
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangles

# Show the image
cv2.imshow("Face Locations", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
