import cv2
import face_recognition
import os
import numpy as np

# Load known images and multiple encodings for each person
images = []
classNames = []

relative_path = 'train'
path = os.path.abspath(relative_path)

mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encoded_face = face_recognition.face_encodings(img)[0]
            encodeList.append(encoded_face)
        except IndexError:
            print("Face encoding failed for an image. Skipping this image.")
    return encodeList

encoded_face_train = findEncodings(images)

# Webcam capture
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)  # Reduce resizing factor for distant faces
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matches = face_recognition.compare_faces(encoded_face_train, encode_face, tolerance=0.5)  # Slightly increase tolerance
        matchIndex = np.argmin(faceDist)

        # Check if the closest match distance is below threshold
        threshold = 0.9
        if faceDist[matchIndex] < threshold and matches[matchIndex]:
            name = classNames[matchIndex].capitalize()
        else:
            name = "Unknown Person"
        
        # Scale face coordinates back
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







