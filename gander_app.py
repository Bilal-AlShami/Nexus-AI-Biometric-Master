import sys
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ZeroPadding2D
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTabWidget, QFrame, QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from deepface import DeepFace

# هذا الأمر سيقوم بتحميل نموذج العمر وتجهيزه تلقائياً
model = DeepFace.build_model("Gander")
# --- Model & Constants ---
IMG_SIZE = 224
MODEL_PATH = 'gender_model.h5'

# Load Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def build_age_model():
    # VGG16-like Fully Convolutional Architecture
    model = Sequential()
    
    # Block 1
    model.add(ZeroPadding2D((1,1), input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Block 2
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Block 3
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Block 4
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Block 5
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Top
    model.add(Conv2D(4096, (7, 7), activation='relu', padding='valid'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu', padding='valid'))
    model.add(Dropout(0.5))
    model.add(Conv2D(101, (1, 1), activation='softmax')) 
    model.add(Flatten())
    
    return model

# --- Worker Thread ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self._run_flag = True
        self.model = build_age_model()
        
        try:
            self.model.load_weights(MODEL_PATH)
            print("Age model loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")

        # Config camera
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        frame_count = 0
        last_results = [] # (x,y,w,h, age)

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret: break
            
            # Optimization: Detect & Predict every 5 frames
            if frame_count % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
                faces = face_cascade.detectMultiScale(small, 1.1, 5)
                
                last_results = []
                for (sx, sy, sw, sh) in faces:
                    x,y,w,h = sx*2, sy*2, sw*2, sh*2
                    
                    face_roi = frame[y:y+h, x:x+w].copy()
                    if face_roi.size == 0: continue
                    
                    # Preprocess for model
                    # VGG Face usually expects resize to 224, and specific mean subtraction
                    # Valid attempt: Resize -> RGB -> Array
                    blob = cv2.resize(face_roi, (224, 224))
                    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
                    blob = np.expand_dims(blob, axis=0)
                    
                    # Simple prediction
                    preds = self.model.predict(blob, verbose=0) 
                    # preds shape (1, 101)
                    
                    # Calculate apparent age: sum(i * prob[i])
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_age = preds[0].dot(ages).flatten()[0]
                    
                    last_results.append((x,y,w,h, predicted_age))
            
            # Draw results
            for (x,y,w,h, age) in last_results:
                label = f"Age: {int(age)}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            self.change_pixmap_signal.emit(frame)
            frame_count += 1
            
        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class AgeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Age Recognition AI")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        main_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        # Header
        title = QLabel("Age Estimation System")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Display
        self.display_label = QLabel("Starting Camera...")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet("background-color: #000; border: 2px solid #333;")
        self.display_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.display_label)
        
        # Button
        self.btn = QPushButton("Stop")
        self.btn.setStyleSheet("background-color: #d32f2f; padding: 10px; font-weight: bold;")
        self.btn.clicked.connect(self.close)
        main_layout.addWidget(self.btn)
        
        # Start Thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.display_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.display_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgeApp()
    window.show()
    sys.exit(app.exec())
