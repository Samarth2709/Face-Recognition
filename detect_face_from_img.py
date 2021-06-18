import cv2

# Pre-trained data from frontal face cv2
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load image
img = cv2.imread("group_photo.jpg")


# Change image to grayscaled
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect faces and get coords
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
for x, y, w, h in face_coordinates:
    # display rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 225, 0), 5)

print(face_coordinates)

# To resize image
win_name = "Image Display"  #  1. use var to specify window name everywhere
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  #  2. use 'normal' flag
h,w = img.shape[:2]  #  suits for image containing any amount of channels
resize_factor = 1
h = int(h / resize_factor)  #  one must compute beforehand
w = int(w / resize_factor)  #  and convert to INT
cv2.resizeWindow(win_name, w, h)  #  use variables defined/computed BEFOREHAND


# To show image
cv2.imshow(win_name, img)
cv2.waitKey()


print("Code Completed")
