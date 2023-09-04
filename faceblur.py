import cv2

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces and apply a blur effect
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) which is the face
        face_roi = frame[y:y + h, x:x + w]

        # Apply a blur effect to the face ROI
        face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)

        # Replace the original face with the blurred face
        frame[y:y + h, x:x + w] = face_roi

    # Display the result
    cv2.imshow('Face Detection and Blur', frame)

    # Press 'q' to exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
