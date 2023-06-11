import cv2
import sys
from mymodel import FaceEmotionsModel
from frame_drawing import FrameWithEmotion

img_path = sys.argv[1]
img = cv2.imread(img_path)

if len(sys.argv) == 3:
    model = FaceEmotionsModel(sys.argv[2])
else:
    model = FaceEmotionsModel()

trained_face_data = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
grayscale_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscale_frame)
draftsman = FrameWithEmotion(model)
for (x, y, w, h) in face_coordinates:
    draftsman.draw_emotion_frame(img, x, y, w, h)

cv2.imshow('Emotion Recognition', img)
while(True):
    k = cv2.waitKey(33)
    if k == -1:
        continue
    else:
        break
cv2.destroyWindow('Emotion Recognition')