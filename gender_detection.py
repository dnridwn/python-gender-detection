import cv2
import numpy as np

# Load the gender detection model
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

# Load the gender labels
gender_labels = ['Wanita', 'Pria']

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    _, frame = cap.read()

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Feed the blob to the gender detection model
    gender_net.setInput(blob)
    predictions = gender_net.forward()

    # Get the gender with the highest probability
    i = np.argmax(predictions[0])
    gender = gender_labels[i]
    gender_confidence = predictions[0][i]

    # Draw the gender label on the frame
    cv2.putText(frame, f'Gender: {gender} ({gender_confidence * 100:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam
cap.release()
