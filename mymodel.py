import tensorflow as tf
from keras.models import load_model
import os
import numpy as np
import cv2
import math

class FaceEmotionsModel(tf.keras.Model):

    def __init__(self, va = ''):
        super().__init__()
        self.va = va
        if self.va == 'va':
            self.model = load_model(os.path.join(os.getcwd(), 'MyVA_model',
            '31_checkpoint_0.058'))
        else:
            self.model = load_model(os.path.join(os.getcwd(), 'My_model',
            '14_checkpoint_0.453'))
        self.names = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear",
             4: "happy", 5: "neutral", 6: "sad", 7: "surprise", 8: "uncertain"}
        self.coords = {'anger': [-0.46, 0.89],
         'contempt': [-0.65, 0.76],
         'disgust': [-0.81, 0.58],
         'fear': [-0.14, 0.99],
         'happy': [0.98, 0.19],
         'neutral': [0, 0],
         'sad': [-0.9, -0.44],
         'surprise': [0.43, 0.90]}
        self.img_size = 224

    def one_emotion_predict_va(self, img, call_request = False):
        names_reverse = {v : str(k) for k, v in self.names.items()}
        img = img.numpy()
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[np.newaxis, ...]
        if call_request:
            pred = self.model(img).numpy()[0]
        else:
            pred = self.model.predict(img)[0]
        dist_neutral = np.sqrt(pred[0] ** 2 + pred[1] ** 2)
        if dist_neutral <= 0.1:
            return ['neutral', pred[0], pred[1], str(1 - dist_neutral)]
        else:
            coords_short = self.coords.copy()
            del coords_short['neutral']
            emo_dist = {}
            for emotion in coords_short:
                p3 = np.array(self.coords[emotion])
                p1 = pred
                p2 = np.array([0, 0])
                emo_dist[emotion] = np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p3))
            emo_dist = dict(sorted(emo_dist.items(), key=lambda item: item[1]))
            if list(emo_dist.values())[0] > 0.1 or np.sqrt(pred[0] ** 2 + pred[1] ** 2) > 1:
                return ['uncertain', pred[0], pred[1], 'no data']
            else:
                angle_pred = math.atan2(pred[1], pred[0])
                angle_emo = math.atan2(self.coords[list(emo_dist.keys())[0]][1], self.coords[list(emo_dist.keys())[0]][0])
                dist_pred = math.sqrt(pred[1] ** 2 + pred[0] ** 2)
                probability = abs(math.cos(abs(angle_pred - angle_emo)) * dist_pred)
                return [list(emo_dist.keys())[0], pred[0], pred[1], str(probability)]

    def one_emotion_predict(self, img, call_request = False):
        img = img.numpy()
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[np.newaxis, ...]
        if call_request:
            pred = self.model(img).numpy()[0]
        else:
            pred = self.model.predict(img)[0]
        return [self.names[np.argmax(pred)], np.max(pred)]

    def call(self, img):
        if self.va == 'va':
            return self.one_emotion_predict_va(img, call_request = True)
        else:
            print(type(img))
            return self.one_emotion_predict(img, call_request = True)
        

    def predict(self, img):
        if self.va == 'va':
            return self.one_emotion_predict_va(img)
        else:
            return self.one_emotion_predict(img)
