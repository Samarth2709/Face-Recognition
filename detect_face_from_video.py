import cv2
import os
import time

def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))


# Load pre-trained data on face
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get Webcam (Default)
webcam = cv2.VideoCapture(0)
avg = 0
total_faces = 0
loop = 0
start_time = time.time()

while True:
    successful_frame_read, frame = webcam.read()

    # Gray Scale image
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face and get coords
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    loop+=1
    total_faces += len(face_coordinates)

    for x, y, w, h in face_coordinates:
        # display rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 5)

    # Display Image
    cv2.imshow("Image Display", frame)
    key = cv2.waitKey(1)

    end_time = time.time()
    if key != -1 and key != 13:
        print(key)
        break
    elif key == 13 or end_time-start_time>2:
        avg = int(total_faces / loop)
        total_faces = 0
        loop = 0
        os.system('cls')
        print("There is an average of", avg, "faces.")

        end_time = time.time()
        start_time = end_time
webcam.release()
