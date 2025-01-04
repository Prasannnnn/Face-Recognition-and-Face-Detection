import face_recognition
import cv2
import os
import numpy as np

# Path to the directory with known images
known_faces_dir = r"C:\Users\ADMIN\Desktop\AIML-Projects\Face-Detection-and-Face-recognition\images"

# Load and encode known faces
known_encodings = []
known_names = []

# Iterate through all files in the known faces directory
for file_name in os.listdir(known_faces_dir):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        # Load the image
        image_path = os.path.join(known_faces_dir, file_name)
        image = face_recognition.load_image_file(image_path)

        # Encode the face
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Ensure at least one face is detected
            known_encodings.append(encodings[0])
            # Extract the name from the filename (e.g., "Alice.png" -> "Alice")
            known_names.append(os.path.splitext(file_name)[0])

# Load the unknown image
unknown_image_path = r"C:\Users\ADMIN\Desktop\AIML-Projects\Face-Detection-and-Face-recognition\unknown.jpeg"
unknown_image = face_recognition.load_image_file(unknown_image_path)

# Detect face locations and encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
unknown_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the unknown image to OpenCV format for display
unknown_image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Iterate through detected faces
for (top, right, bottom, left), unknown_encoding in zip(face_locations, unknown_encodings):
    # Calculate distances from known encodings
    distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(distances)

    # Visualize distances (for debugging)
    print(f"Distances: {distances}")
    print(f"Best Match Index: {best_match_index}")

    # Determine the name
    tolerance = 0.4  # Adjust as needed (default is 0.6)
    if distances[best_match_index] < tolerance:
        name = known_names[best_match_index]
    else:
        name = "Unknown"

    # Draw a rectangle around the face
    cv2.rectangle(unknown_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

    # Annotate the name below the face
    cv2.putText(unknown_image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Add confidence score to the annotation
    confidence = 1 - distances[best_match_index]  # Higher confidence is better
    cv2.putText(unknown_image_bgr, f"{confidence:.2f}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Display the annotated image
cv2.imshow("Face Recognition", unknown_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
