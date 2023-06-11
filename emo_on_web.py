import cv2
import sys
from mymodel import FaceEmotionsModel
from frame_drawing import FrameWithEmotion

vid = cv2.VideoCapture(0)

if len(sys.argv) == 2:
    model = FaceEmotionsModel(sys.argv[1])
else:
    model = FaceEmotionsModel()
draftsman = FrameWithEmotion(model)

trained_face_data = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
while True:
	ret, frame = vid.read()
	if not ret:
		break
	grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face_coordinates = trained_face_data.detectMultiScale(grayscale_frame)
	for (x, y, w, h) in face_coordinates:
		draftsman.draw_emotion_frame(frame, x, y, w, h)
	cv2.imshow('Video with emotion', frame)
	k = cv2.waitKey(1)
	if k == -1:  # if no key was pressed, -1 is returned
		continue
	else:
		break
vid.release()
cv2.destroyWindow('Video with emotion')