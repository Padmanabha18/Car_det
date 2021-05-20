import cv2 as cv
cap = cv.VideoCapture('car.mp4')
car = cv.CascadeClassifier('car.xml')
while True:
    ret, frames = cap.read()
    gray = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
    cars = car.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in cars:
        cv.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.imshow('video2', frames)
        if cv.waitKey(33) == 27:
            break
cv.destroyWindow()

