import streamlit as st
import cv2
import numpy as np
import face_recognition
from PIL import Image
import pickle

# Load known faces from the model
def load_model():
    with open('face_encodings_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data['encodings'], model_data['names']

# Function to recognize faces in an uploaded image
def recognize_faces_in_image(uploaded_image):
    # Load the image and convert it to RGB
    image = np.array(uploaded_image.convert('RGB'))

    # Detect face locations and encodings in the uploaded image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    known_encodings, known_names = load_model()

    # Create an OpenCV image for visualization
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process each face in the image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate the distances between known encodings and the face
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        # Determine the name of the face
        tolerance = 0.4
        if distances[best_match_index] < tolerance:
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        # Draw a rectangle around the face and add name text
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert the image back to RGB for display in Streamlit
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# Function to access webcam and predict faces in real-time
def real_time_face_recognition():
    # Start the webcam capture
    cap = cv2.VideoCapture(0)

    known_encodings, known_names = load_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (OpenCV uses BGR by default)
        rgb_frame = frame[:, :, ::-1]

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Draw rectangles and labels for faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Calculate the distances between known encodings and the face
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)

            # Determine the name of the face
            tolerance = 0.4
            if distances[best_match_index] < tolerance:
                name = known_names[best_match_index]
            else:
                name = "Unknown"

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the frame with face recognition
        cv2.imshow("Real-time Face Recognition", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit Interface
st.title("Face Recognition with Webcam & Image Upload")

# Option to choose between webcam or image upload
option = st.radio("Choose mode", ("Webcam", "Upload Image"))

if option == "Webcam":
    st.write("Real-time face recognition using your webcam:")
    st.text("Press 'q' to stop the webcam feed")
    real_time_face_recognition()  # Start webcam face recognition

elif option == "Upload Image":
    st.write("Upload an image for face recognition:")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Process the uploaded image
        uploaded_image = Image.open(uploaded_file)
        result_image = recognize_faces_in_image(uploaded_image)
        
        # Display the result
        st.image(result_image, caption="Processed Image", use_column_width=True)
