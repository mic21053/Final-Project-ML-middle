import cv2

class FrameWithEmotion():

	def __init__(self, model):
		self.model = model

	def draw_emotion_frame(self, img, x, y, w, h):
		face_in_img = img[y:y + h, x:x + w]
		pred = self.model(face_in_img)
		bad_emotions = ['anger', 'contempt', 'disgust', 'fear', 'sad']
		good_emotion = ['happy', 'surprise']
		neutral_emotions = ['neutral', 'uncertain']
		if pred[0] in bad_emotions:
			rectangle_color = (0, 0, 255)
		elif pred[0] in good_emotion:
			rectangle_color = (0, 255, 0)
		else:
			rectangle_color = (128, 128, 128)
		if pred[0] == 'anger':
			font_color = (255, 0, 0)
		elif pred[0] == 'contempt':
			font_color = (255, 0, 127)
		elif pred[0] == 'disgust':
			font_color = (0, 76, 153)
		elif pred[0] == 'fear':
			font_color = (76, 0, 153)
		elif pred[0] == 'happy':
			font_color = (0, 0, 128)
		elif pred[0] == 'neutral':
			font_color = (128, 128, 128)
		elif pred[0] == 'sad':
			font_color = (255, 0, 255)
		elif pred[0] == 'surprise':
			font_color = (255, 153, 255)
		else:
			font_color = (0, 0, 0)
		font_name = 16
		font_scale = 0.45
		font_thickness = 1
		font_line_type = cv2.LINE_AA
		label_org = (x + 2, y + 15)
		if len(pred) == 2:
			emo_name = f'{pred[0]} ({int(pred[1] * 100)}%)'
		else:
			if pred[3] == 'no data':
				emo_name = f'{pred[0]} (no data) V = {pred[1]:.2f} A = {pred[2]:.2f}'
			else:
				emo_name = f'{pred[0]} ({int(float(pred[3]) * 100)}%) V = {pred[1]:.2f} A = {pred[2]:.2f}'
		cv2.rectangle(img, (x, y), (x + w, y + h), rectangle_color, thickness=2)
		cv2.rectangle(img, (x, y), (x + 300, y + 20), (255, 255, 255), thickness=-1)
		cv2.putText(img,
                text=emo_name,
                org=label_org,
                fontFace=font_name,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness,
                lineType=font_line_type)
		return img
			
			