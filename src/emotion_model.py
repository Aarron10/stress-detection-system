import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

class EmotionModel:
    def __init__(self, model_path='model.h5'):
        self.model = self.build_model()
        self.load_weights(model_path)
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    def build_model(self):
        # Neural Network Architecture (Same as kerasmodel.py)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        return model

    def load_weights(self, path):
        if os.path.exists(path):
            self.model.load_weights(path)
            print(f"Emotion Model loaded from {path}")
        else:
            print(f"Error: {path} not found.")

    def predict(self, face_crop):
        """
        Takes a BGR face crop, preprocesses it, and returns emotion probabilities.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # Preprocessing suitable for this model:
        # 1. Grayscale
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # 1b. Histogram Equalization (Lighting Normalization)
        gray = cv2.equalizeHist(gray)
        
        # 2. Resize to 48x48
        resized = cv2.resize(gray, (48, 48))
        # 3. Expand dims to (1, 48, 48, 1)
        input_data = np.expand_dims(np.expand_dims(resized, -1), 0)
        
        # Predict
        preds = self.model.predict(input_data, verbose=0)[0]
        
        # Return dict: {'Angry': 0.1, 'Happy': 0.9 ...}
        return {self.emotion_dict[i]: float(preds[i]) for i in range(7)}
