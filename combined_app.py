import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import deque, Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ZeroPadding2D
import random
import time

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,QScrollArea, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QSizePolicy, QGraphicsDropShadowEffect,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QLinearGradient, QBrush, QPen, QPalette, QTransform, QPainterPath
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QPoint, QRectF, QRect
import math
import sqlite3
from datetime import datetime
# MediaPipe removed - using OpenCV DNN instead
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

# Try importing TTS but don't fail if missing
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
IMG_SIZE_TORCH = 380
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTION_CLASSES = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']

# --- SHARED STATE ---
class AppState:
    def __init__(self):
        self.mutex = QMutex()
        self.latest_frame = None
        self.latest_results = []
        self.dominant_emotion = "Neutral"
        self.is_running = True
        self.is_running = True
        self.theme_mode = "Night"
        self.session_snapshots = []
        self.emotion_counts = {emo: 0 for emo in EMOTION_CLASSES}
        self.vision_mode = "Normal" # "Normal", "Matrix", "Thermal"
        self.intensity = 0.0 # Neural intensity (0 to 1)
        self.last_voice_time = 0
        self.threat_level = 0.0 # 0 to 1
        self.security_logs = deque(maxlen=20)

shared_state = AppState()

# --- DATABASE MANAGER ---
class DatabaseManager:
    def __init__(self, db_path="nexus_biometrics.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                identity TEXT,
                gender TEXT,
                age INTEGER,
                emotion TEXT,
                focus_score REAL,
                gesture TEXT,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_record(self, data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO records (timestamp, identity, gender, age, emotion, focus_score, gesture, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            data.get('identity', 'Unknown'),
            data.get('gender', 'Unknown'),
            data.get('age', 0),
            data.get('emotion', 'Neutral'),
            data.get('focus_score', 0.0),
            data.get('gesture', 'None'),
            data.get('image_path', '')
        ))
        conn.commit()
        conn.close()

db_manager = DatabaseManager()

# --- MODEL CONSTRUCTORS ---

def get_emotion_model():
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(EMOTION_CLASSES))
    if os.path.exists('best_model_colab.pth'):
        try:
            model.load_state_dict(torch.load('best_model_colab.pth', map_location=DEVICE, weights_only=True))
            print("INFO: Emotion model loaded.")
        except Exception as e:
            print(f"WARNING: No emotion weights found or loading error: {e}")
    model = model.to(DEVICE).eval()
    return model

def build_vgg_custom(num_classes):
    """Keras VGG-16 for Age/Gender (Full Structure to avoid OOM)"""
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
    model.add(Conv2D(num_classes, (1, 1), activation='softmax'))
    model.add(Flatten())
    return model

# Paths
EMOTION_MODEL_PATH = 'best_model_colab.pth'
AGE_MODEL_PATH = 'age_model.h5'
GENDER_MODEL_PATH = 'gender_model.h5'

# --- MODEL LOADERS ---

def get_age_model():
    model = build_vgg_custom(101) # 101 Classes
    if os.path.exists(AGE_MODEL_PATH):
        try: model.load_weights(AGE_MODEL_PATH)
        except: print("WARNING: Age weights failed.")
    return model

def get_gender_model():
    model = build_vgg_custom(2) # Binary
    if os.path.exists(GENDER_MODEL_PATH):
        try: model.load_weights(GENDER_MODEL_PATH)
        except: 
            # Fallback if 101 class
            model = build_vgg_custom(101)
            try: model.load_weights(GENDER_MODEL_PATH)
            except: print("WARNING: Gender weights failed.")
    return model

def get_head_pose(x, y, w, h, frame_w, frame_h):
    # Simplified Gaze: Distance from center
    cx, cy = x + w/2, y + h/2
    off_x = (cx - frame_w/2) / (frame_w/2)
    off_y = (cy - frame_h/2) / (frame_h/2)
    
    yaw = off_x * 45  # Horizontal
    pitch = off_y * 30 # Vertical
    
    status = "FOCUSED"
    if abs(yaw) > 15 or abs(pitch) > 15: status = "DISTRACTED"
    return status, yaw, pitch

# --- WORKER THREADS ---

class CameraWorker(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while shared_state.is_running:
            ret, frame = cap.read()
            if ret:
                shared_state.mutex.lock()
                shared_state.latest_frame = frame.copy()
                shared_state.mutex.unlock()
                self.frame_signal.emit(frame)
            time.sleep(0.01) # Maintain ~60 FPS
        cap.release()

class AIWorker(QThread):
    status_signal = pyqtSignal(str)
    greeting_signal = pyqtSignal(str, int, str) # Label, Age, Gender
    log_signal = pyqtSignal(str)
    gesture_signal = pyqtSignal(str) # Gesture detected

    def __init__(self):
        super().__init__()
        self.face_buffers = {} # {id: deque}
        self.next_face_id = 0

    def run(self):
        # Load Models
        self.status_signal.emit("LOADING AI CORES...")
        e_model = get_emotion_model()
        a_model = get_age_model()
        g_model = get_gender_model()
        
        # Face Recognition Setup
        faces_dir = "faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
        
        # Load OpenCV Haar Cascade face detector
        self.status_signal.emit("LOADING FACE DETECTOR...")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.status_signal.emit("SYSTEM: ONLINE")
        trans = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        while shared_state.is_running:
            frame = None
            shared_state.mutex.lock()
            if shared_state.latest_frame is not None:
                frame = shared_state.latest_frame.copy()
            shared_state.mutex.unlock()
            
            if frame is None:
                time.sleep(0.1)
                continue

            fh, fw = frame.shape[:2]
            
            # OpenCV Haar Cascade Face Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            
            current_results = []
            emos_summary = []
            new_buffers = {}
            gesture = "None"  # Gesture control disabled without MediaPipe Hands
            
            for (x, y, w, h) in faces:
                
                # Ensure ROI is within frame
                x, y = max(0, x), max(0, y)
                w, h = min(w, fw - x), min(h, fh - y)
                
                if w < 40 or h < 40: continue
                
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0: continue
                
                # Preprocessing for better accuracy
                g_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                eq_roi = clahe.apply(g_roi)
                proc_roi = cv2.cvtColor(eq_roi, cv2.COLOR_GRAY2RGB)
                
                # 1. Emotion (EfficientNet)
                try:
                    pil = Image.fromarray(proc_roi)
                    tin = trans(pil).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = e_model(tin)
                        probs = torch.nn.functional.softmax(out, dim=1)
                        conf, pred = torch.max(probs, 1)
                    
                    # Tracking logic
                    matched_id = None
                    min_dist = 100
                    for fid, buf_info in self.face_buffers.items():
                        px, py = buf_info['last_pos']
                        dist = np.sqrt((x-px)**2 + (y-py)**2)
                        if dist < min_dist:
                            min_dist = dist
                            matched_id = fid
                    
                    if matched_id is None:
                        matched_id = self.next_face_id
                        self.next_face_id += 1
                        new_buffers[matched_id] = {'queue': deque(maxlen=15), 'last_pos': (x,y)}
                    else:
                        new_buffers[matched_id] = self.face_buffers[matched_id]
                        new_buffers[matched_id]['last_pos'] = (x,y)
                    
                    new_buffers[matched_id]['queue'].append(pred.item())
                    smooth_pred = Counter(new_buffers[matched_id]['queue']).most_common(1)[0][0]
                    label = EMOTION_CLASSES[smooth_pred]
                except Exception as e:
                    print(f"DEBUG AI: {e}")
                    label = "Neutral"
                    probs = torch.zeros((1, len(EMOTION_CLASSES)))
                
                emos_summary.append(label)

                # 2. Age/Gender
                try:
                    k_roi = cv2.resize(roi, (224, 224))
                    k_roi = cv2.cvtColor(k_roi, cv2.COLOR_BGR2RGB)
                    k_in = np.expand_dims(k_roi, axis=0) / 255.0
                    
                    age_p = a_model.predict(k_in, verbose=0)
                    gen_p = g_model.predict(k_in, verbose=0)
                    
                    age = int(age_p[0].dot(np.arange(0, 101)))
                    g_idx = np.argmax(gen_p[0])
                    if gen_p.shape[1] > 2:
                        gender = "Male" if g_idx < 50 else "Female"
                    else:
                        gender = "Male" if g_idx == 1 else "Female"
                        
                    age_conf = float(np.max(age_p[0]))
                    gen_conf = float(np.max(gen_p[0]))
                except Exception as e:
                    print(f"CORE AI ERROR: {e}")
                    age, gender = 25, "Calibrating..."
                    age_conf, gen_conf = 0.5, 0.5

                # 3. Identity (DeepFace)
                identity = "Unknown"
                if HAS_DEEPFACE and os.listdir(faces_dir):
                    try:
                        # Perform recognition only every few frames to save CPU
                        if matched_id % 5 == 0: 
                            results_df = DeepFace.find(img_path=roi, db_path=faces_dir, enforce_detection=False, silent=True)
                            if results_df and not results_df[0].empty:
                                # Get the name from the filename
                                identity = os.path.basename(results_df[0]['identity'][0]).split('.')[0]
                    except: pass

                # 4. Focus/Pose using simple geometry (fallback without MediaPipe)
                # Calculate face center relative to frame center
                face_cx = x + w // 2
                face_cy = y + h // 2
                frame_cx, frame_cy = fw // 2, fh // 2
                
                yaw = (face_cx - frame_cx) / (fw / 2) * 30  # -30 to 30 degrees
                pitch = (face_cy - frame_cy) / (fh / 2) * 20  # -20 to 20 degrees
                
                focus_status = "FOCUSED"
                if abs(yaw) > 15 or abs(pitch) > 15: focus_status = "DISTRACTED"

                res_data = {
                    'bbox': (x, y, w, h),
                    'identity': identity,
                    'emotion': label,
                    'emotion_probs': probs[0].tolist(),
                    'age': age,
                    'age_conf': age_conf,
                    'gender': gender,
                    'gen_conf': gen_conf,
                    'focus': focus_status,
                    'yaw': yaw,
                    'pitch': pitch,
                    'gesture': gesture
                }
                current_results.append(res_data)

                # Threat Calculation
                shared_state.mutex.lock()
                if "Angry" in label or "Fear" in label:
                    shared_state.threat_level = min(1.0, shared_state.threat_level + 0.15)
                else:
                    shared_state.threat_level = max(0.0, shared_state.threat_level - 0.05)
                shared_state.mutex.unlock()

                # Trigger Voice Greeting
                now = time.time()
                if now - shared_state.last_voice_time > 10:
                    triggers = ["Happy", "Angry", "Sad", "Surprise"]
                    if any(t in label for t in triggers):
                        shared_state.last_voice_time = now
                        self.greeting_signal.emit(label, age, gender)
                    self.log_signal.emit(f"TRACKING_{gender[0]}:{age} - {label.upper()}")

            self.face_buffers = new_buffers 

            # --- UPDATE SHARED STATE (LOCKED) ---
            shared_state.mutex.lock()
            shared_state.latest_results = current_results # CRITICAL FIX: Update UI results
            
            if emos_summary: 
                dom = Counter(emos_summary).most_common(1)[0][0]
                shared_state.dominant_emotion = dom
                shared_state.emotion_counts[dom] += 1
                # Update intensity based on emotion
                if any(x in dom for x in ["Surprise", "Angry", "Fear"]):
                    shared_state.intensity = min(1.0, shared_state.intensity + 0.2)
                else:
                    shared_state.intensity = max(0.1, shared_state.intensity - 0.05)
            else:
                shared_state.intensity = max(0.05, shared_state.intensity - 0.02)
            shared_state.mutex.unlock()
            
            time.sleep(0.01) # Ultra-fast response

class VoiceWorker(QThread):
    command_signal = pyqtSignal(str)

    def run(self):
        if not HAS_SR: return
        
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        while shared_state.is_running:
            try:
                with microphone as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
                
                text = recognizer.recognize_google(audio).lower()
                if "nexus" in text:
                    if "screenshot" in text or "capture" in text:
                        self.command_signal.emit("SNAPSHOT")
                    elif "night" in text or "dark" in text or "day" in text or "light" in text:
                        self.command_signal.emit("TOGGLE_THEME")
                    elif "vision" in text or "cycle" in text:
                        self.command_signal.emit("CYCLE_VISION")
                    elif "terminate" in text or "exit" in text:
                        self.command_signal.emit("EXIT")
            except:
                pass
            time.sleep(0.1)

# --- LUXURY COMPONENTS ---

class FaceIDCard(QFrame):
    def __init__(self, face_img, data):
        super().__init__()
        self.setFixedWidth(270)
        self.setFixedHeight(140)
        
        is_night = (shared_state.theme_mode == "Night")
        accent = "#00f2ff" if is_night else "#c5a000"
        
        bg_color = "rgba(20, 20, 30, 0.6)" if is_night else "rgba(240, 240, 245, 0.8)"
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {accent}44;
                border-radius: 12px;
            }}
        """)
        
        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(10, 10, 10, 10)
        main_lay.setSpacing(8)
        
        top_lay = QHBoxLayout()
        # Snapshot
        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(70, 70)
        pix = QPixmap.fromImage(face_img).scaled(70, 70, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        self.img_lbl.setPixmap(pix)
        self.img_lbl.setStyleSheet(f"border-radius: 8px; border: 1.5px solid {accent};")
        
        # Info
        info_lay = QVBoxLayout()
        name_val = data.get('identity', data['gender']).upper()
        name = QLabel(f"ID: {name_val}")
        name.setStyleSheet(f"color: {accent}; font-weight: 900; font-size: 11px; letter-spacing: 1px;")
        
        stats = QLabel(f"AGE: {data['age']} | {data['emotion'].split()[0].upper()}")
        stats.setStyleSheet(f"color: {'#fff' if is_night else '#222'}; font-size: 10px; font-weight: bold;")
        
        focus = QLabel(f"STANCE: {data['focus']}")
        focus_col = '#00ff9d' if 'FOCUS' in data['focus'] else '#ff5500'
        focus.setStyleSheet(f"color: {focus_col}; font-size: 9px; font-weight: bold;")
        
        info_lay.addWidget(name)
        info_lay.addWidget(stats)
        info_lay.addWidget(focus)
        
        top_lay.addWidget(self.img_lbl)
        top_lay.addLayout(info_lay)
        
        # Emotion Frequency Bar (Mini Spectrum)
        bar_lay = QHBoxLayout()
        bar_lay.setSpacing(2)
        probs = data.get('emotion_probs', [0.1]*7) # Fallback
        
        for p in probs:
            bar = QFrame()
            bar.setFixedHeight(max(2, int(p * 30)))
            bar.setFixedWidth(28)
            bar.setStyleSheet(f"background: {accent}; border-radius: 1px; opacity: 0.8;")
            bar_lay.addWidget(bar, alignment=Qt.AlignmentFlag.AlignBottom)
            
        main_lay.addLayout(top_lay)
        main_lay.addLayout(bar_lay)

class DataStreamWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(240)
        self.setFixedHeight(120)
        self.lines = deque(maxlen=6)
        
        lay = QVBoxLayout(self)
        self.lbl = QLabel("SYSTEM READY")
        self.lbl.setFont(QFont("Consolas", 8))
        self.lbl.setStyleSheet("color: #00ff9d;")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)

    def add_log(self, text):
        self.lines.append(f"> {text}")
        self.lbl.setText("\n".join(self.lines))

class AnalyticsChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(180)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(500)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        shared_state.mutex.lock()
        counts = dict(shared_state.emotion_counts)
        shared_state.mutex.unlock()
        
        total = sum(counts.values()) or 1
        is_night = (shared_state.theme_mode == "Night")
        accent = QColor(0, 242, 255) if is_night else QColor(197, 160, 0)
        
        w, h = self.width(), self.height()
        bar_w = (w - 40) // len(EMOTION_CLASSES)
        
        for i, (emo, count) in enumerate(counts.items()):
            x = 20 + i * bar_w
            # Map count to height
            perc = count / total
            bar_h = int(perc * (h - 40))
            
            # Glow effect
            painter.setOpacity(0.2)
            painter.setBrush(accent)
            painter.drawRect(x+5, h-20-bar_h, bar_w-10, bar_h)
            
            # Main Bar
            painter.setOpacity(1.0)
            painter.setPen(QPen(accent, 1))
            painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 150))
            painter.drawRoundedRect(x+5, h-20-bar_h, bar_w-10, bar_h, 3, 3)
            
            # Icon/Name tiny
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(x, h-5, emo.split()[-1])

class NeuralWaveform(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(60)
        self.phase = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_wave)
        self.timer.start(30)

    def update_wave(self):
        self.phase += 0.2
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        is_night = (shared_state.theme_mode == "Night")
        accent = QColor(0, 242, 255) if is_night else QColor(197, 160, 0)
        
        # Intensity affects frequency and amplitude
        shared_state.mutex.lock()
        inten = shared_state.intensity
        shared_state.mutex.unlock()
        
        painter.setPen(QPen(accent, 2))
        path = QPainterPath()
        mid_y = self.height() // 2
        
        for x in range(0, self.width(), 2):
            # Complex wave
            y = mid_y + math.sin(x * 0.05 + self.phase) * (10 + 30 * inten)
            y += math.sin(x * 0.1 - self.phase * 0.5) * (5 * inten)
            if x == 0: path.moveTo(x, y)
            else: path.lineTo(x, y)
            
        painter.drawPath(path)

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, w, h, emotion):
        if "Happy" in emotion: color = (255, 215, 0) # Gold
        elif "Angry" in emotion: color = (255, 50, 50) # Red
        else: color = (0, 242, 255) # Cyan
        
        for _ in range(2):
            self.particles.append([
                random.randint(x, x+w), random.randint(y, y+h), # pos
                random.uniform(-1, 1), random.uniform(-3, -1), # vel
                random.randint(30, 60), color, random.randint(2, 4) # life, col, size
            ])

    def update_and_draw(self, painter):
        alive = []
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
            if p[4] > 0:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(*p[5], int(p[4] * 4)))
                painter.drawEllipse(QPoint(int(p[0]), int(p[1])), p[6], p[6])
                alive.append(p)
        self.particles = alive

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.particles = ParticleSystem()
        self.scan_pos = 0
        self.shutter_flash = 0
        self.setMouseTracking(True)
        self.tilt_x = 0
        self.tilt_y = 0

    def mouseMoveEvent(self, event):
        # Smoother Parallax effect
        cx, cy = self.width() / 2, self.height() / 2
        target_tilt_x = (event.position().x() - cx) / cx * 10
        target_tilt_y = (event.position().y() - cy) / cy * -10
        
        # Simple lerp for smoothness
        self.tilt_x += (target_tilt_x - self.tilt_x) * 0.1
        self.tilt_y += (target_tilt_y - self.tilt_y) * 0.1
        self.update()

    def update_frame(self, frame):
        self.frame = frame
        self.update()

    def apply_vision(self, frame):
        mode = shared_state.vision_mode
        if mode == "Matrix":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            green = np.zeros_like(frame)
            green[:,:,1] = gray # Green Channel
            # Grid effect
            green[::4, :, 1] = 0
            return green
        elif mode == "Thermal":
            return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        return frame

    def draw_data_block(self, painter, x, y, title, value, color):
        """Draws a right-aligned techy data box for the HUD"""
        w, h = 100, 40
        rect = QRect(x, y, w, h)
        
        # Backdrop
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(rect, 3, 3)
        
        # Glow Border (Right side)
        painter.setPen(QPen(color, 2))
        painter.drawLine(x + w, y, x + w, y + h)
        
        # Text
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(x + 5, y + 12, title)
        
        painter.setPen(color)
        painter.setFont(QFont("Segoe UI Black", 10))
        painter.drawText(x + 5, y + 32, value)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # 3D Tilt Transformation
        transform = QTransform()
        transform.translate(self.width()/2, self.height()/2)
        transform.rotate(self.tilt_x, Qt.Axis.YAxis)
        transform.rotate(self.tilt_y, Qt.Axis.XAxis)
        transform.translate(-self.width()/2, -self.height()/2)
        painter.setTransform(transform)
        
        # Background
        threat_flash = 0
        shared_state.mutex.lock()
        if shared_state.threat_level > 0.7:
            threat_flash = int(100 * math.sin(time.time() * 10))
        shared_state.mutex.unlock()
        
        painter.fillRect(self.rect(), QColor(max(10, threat_flash), 10, 15))
        
        if self.frame is not None:
            proc_frame = self.apply_vision(self.frame)
            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            img = QImage(rgb.data, w, h, c*w, QImage.Format.Format_RGB888)
            scaled = img.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            # Offset
            ox = (self.width() - scaled.width()) // 2
            oy = (self.height() - scaled.height()) // 2
            painter.drawImage(ox, oy, scaled)
            
            # Mapping factor
            fx = scaled.width() / w
            fy = scaled.height() / h
            
            # HUD Overlay
            shared_state.mutex.lock()
            results = list(shared_state.latest_results)
            shared_state.mutex.unlock()
            
            self.scan_pos = (self.scan_pos + 2) % 100
            
            for res in results:
                bx, by, bw, bh = res['bbox']
                sx, sy = ox + int(bx*fx), oy + int(by*fy)
                sw, sh = int(bw*fx), int(bh*fy)
                
                # Emit Particles
                self.particles.emit(sx, sy, sw, sh, res['emotion'])
                
                is_night = (shared_state.theme_mode == "Night")
                main_col = QColor(0, 242, 255) if is_night else QColor(197, 160, 0)
                
                shared_state.mutex.lock()
                is_alert = shared_state.threat_level > 0.7
                shared_state.mutex.unlock()
                if is_alert: main_col = QColor(255, 50, 50) # RED ALERT
                
                # --- ULTIMATE PIXEL-PERFECT HUD (Phase 13) ---
                center = QPoint(sx + sw//2, sy + sh//2)
                rad_base = max(sw, sh) // 2 + 20
                
                # 1. Multi-Layer Concentric Rotators
                painter.save()
                painter.translate(center)
                
                # Inner Static Circle
                painter.setPen(QPen(main_col, 1))
                painter.drawEllipse(QPoint(0,0), rad_base - 10, rad_base - 10)
                
                # Middle Rotating Dash (Fast)
                painter.rotate(time.time() * 80)
                painter.setPen(QPen(main_col, 2, Qt.PenStyle.DashLine))
                painter.drawEllipse(QPoint(0,0), rad_base, rad_base)
                
                # Outer Rotating Slow (Counter-clockwise)
                painter.rotate(-time.time() * 40)
                painter.setPen(QPen(main_col, 0.5, Qt.PenStyle.DotLine))
                painter.drawEllipse(QPoint(0,0), rad_base + 15, rad_base + 15)
                
                # Crosshairs/Ticks
                painter.setPen(QPen(main_col, 1))
                painter.drawLine(-rad_base, 0, -rad_base+10, 0)
                painter.drawLine(rad_base, 0, rad_base-10, 0)
                painter.drawLine(0, -rad_base, 0, -rad_base+10)
                painter.drawLine(0, rad_base, 0, rad_base-10)
                painter.restore()

                # 2. Main Frame & Brackets (Matching Mockup)
                painter.setPen(QPen(main_col, 3))
                bl = int(sw * 0.25)
                # Outer corners
                off = 20
                painter.drawLine(sx-off, sy-off, sx+bl, sy-off)
                painter.drawLine(sx-off, sy-off, sx-off, sy+bl)
                painter.drawLine(sx+sw+off, sy-off, sx+sw-bl+off, sy-off)
                painter.drawLine(sx+sw+off, sy-off, sx+sw+off, sy+bl)
                # Bottom corners
                painter.drawLine(sx-off, sy+sh+off, sx+bl, sy+sh+off)
                painter.drawLine(sx-off, sy+sh+off, sx-off, sy+sh-bl+off)
                painter.drawLine(sx+sw+off, sy+sh+off, sx+sw-bl+off, sy+sh+off)
                painter.drawLine(sx+sw+off, sy+sh+off, sx+sw+off, sy+sh-bl+off)

                # 3. Branding Header & Decorative Corner Metadata
                painter.setFont(QFont("Segoe UI Black", 16))
                painter.setPen(main_col)
                painter.drawText(QRect(sx, sy - 90, sw, 40), Qt.AlignmentFlag.AlignCenter, "NEXUS")
                
                painter.setFont(QFont("Consolas", 8))
                painter.drawText(sx - 20, sy - 35, "SCANNER")
                painter.drawText(sx + sw - 60, sy - 35, "NEXUS ID BAR GO")
                
                # Battery/Status Icons (Right Top)
                painter.setPen(QPen(main_col, 1))
                painter.drawRect(sx + sw + 5, sy - 85, 30, 14) # Battery outline
                painter.fillRect(sx + sw + 7, sy - 83, 18, 10, main_col) # Charge
                painter.drawText(sx + sw + 40, sy - 74, "4F")

                # 4. Right-Aligned Data Blocks (Consistent with Mockup)
                block_x = sx + sw + 45
                display_name = res.get('identity', res['gender']).upper()
                self.draw_data_block(painter, block_x, sy - 55, "IDENTITY", display_name, main_col)
                self.draw_data_block(painter, block_x, sy, "AGE", str(res['age']), main_col)
                self.draw_data_block(painter, block_x, sy + 55, "EMOTION", res['emotion'].upper(), main_col)
                self.draw_data_block(painter, block_x, sy + 110, "GENDER", res['gender'].upper(), main_col)
                
                # 5. Security Status Bar (Bottom)
                painter.setFont(QFont("Segoe UI Black", 11))
                status_rect = QRect(sx - 20, sy + sh + 40, sw + 40, 55)
                painter.setBrush(QColor(0, 40, 40, 200) if not is_alert else QColor(100, 0, 0, 200))
                painter.setPen(QPen(main_col, 2))
                painter.drawRect(status_rect)
                
                status_txt = "IDENTITY: CONFIRMED - NEXUS ID 942-A"
                if is_alert: status_txt = "THREAT DETECTED - SYSTEM LOCK"
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(status_rect, Qt.AlignmentFlag.AlignCenter, status_txt)

                # Tiny details inside status
                painter.setFont(QFont("Consolas", 7))
                painter.drawText(sx - 10, sy + sh + 55, "BLOCK")
                
                # Scan Line (Laser)
                ly = sy + int((sh * self.scan_pos) / 100)
                painter.setPen(QPen(main_col, 2))
                painter.drawLine(sx-35, ly, sx+sw+35, ly)

        # Shutter Flash Effect
        if self.shutter_flash > 0:
            painter.fillRect(self.rect(), QColor(255, 255, 255, self.shutter_flash))
            self.shutter_flash = max(0, self.shutter_flash - 40)
            QTimer.singleShot(30, self.update)

        self.particles.update_and_draw(painter)

# --- MAIN APPLICATION ---

class NexusLuxuryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEXUS // BIOMETRIC MASTER")
        self.resize(1500, 900)
        
        self.cam_thread = CameraWorker()
        self.ai_thread = AIWorker()
        self.voice_thread = VoiceWorker()
        
        self.init_ui()
        
        self.cam_thread.frame_signal.connect(self.display.update_frame)
        self.ai_thread.status_signal.connect(self.update_ai_status)
        self.ai_thread.greeting_signal.connect(self.handle_greeting)
        self.ai_thread.log_signal.connect(self.stream.add_log) # Connected Logs
        self.ai_thread.gesture_signal.connect(self.handle_gesture)
        self.voice_thread.command_signal.connect(self.handle_voice_command)

        self.cam_thread.start()
        self.ai_thread.start()
        self.voice_thread.start()
        
        # Gallery Timer
        self.gallery_timer = QTimer()
        self.gallery_timer.timeout.connect(self.update_gallery)
        self.gallery_timer.start(2000)

    def update_ai_status(self, text):
        self.lbl_status.setText(text)

    def update_gallery(self):
        shared_state.mutex.lock()
        results = list(shared_state.latest_results)
        frame = None
        if shared_state.latest_frame is not None:
            frame = shared_state.latest_frame.copy()
        shared_state.mutex.unlock()
        
        if results and frame is not None:
            r = results[0]
            bx, by, bw, bh = r['bbox']
            roi = frame[max(0, by):by+bh, max(0, bx):bx+bw]
            if roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                h, w, c = roi_rgb.shape
                qimg = QImage(roi_rgb.data, w, h, c*w, QImage.Format.Format_RGB888).copy()
                
                card = FaceIDCard(qimg, r)
                self.gallery_layout.insertWidget(0, card)
                
                # Feedback: Shutter Flash
                self.display.shutter_flash = 200
                self.display.update()
                
                if self.gallery_layout.count() > 10:
                    item = self.gallery_layout.takeAt(10)
                    if item.widget(): item.widget().deleteLater()

                # --- PHASE 10: DISK ARCHIVING ---
                self.archive_snapshot(roi, r)

    def archive_snapshot(self, roi_bgr, data):
        """Saves face snapshot with HUD overlay and metadata to biometric_archive folder and database"""
        try:
            folder = "biometric_archive"
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"FACE_{data['gender'][0]}_{data['age']}_{timestamp}"
            img_path = os.path.join(folder, f"{filename}.jpg")
            
            # --- RENDER HUD ON IMAGE ---
            canvas = roi_bgr.copy()
            h, w = canvas.shape[:2]
            
            is_night = (shared_state.theme_mode == "Night")
            accent_bgr = (255, 242, 0) if is_night else (0, 160, 197) # BGR
            
            # Corner Brackets
            l = int(w * 0.15)
            cv2.line(canvas, (0, 0), (l, 0), accent_bgr, 2)
            cv2.line(canvas, (0, 0), (0, l), accent_bgr, 2)
            cv2.line(canvas, (w-1, 0), (w-l-1, 0), accent_bgr, 2)
            cv2.line(canvas, (w-1, 0), (w-1, l), accent_bgr, 2)
            cv2.line(canvas, (0, h-1), (l, h-1), accent_bgr, 2)
            cv2.line(canvas, (0, h-1), (0, h-l-1), accent_bgr, 2)
            cv2.line(canvas, (w-1, h-1), (w-l-1, h-1), accent_bgr, 2)
            cv2.line(canvas, (w-1, h-1), (w-1, h-l-1), accent_bgr, 2)

            # Glassmorphic Data Box
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

            # Biometric Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, f"DEEP TRACKING: {data['gender'][0]}_{data['age']}", (10, 15), font, 0.4, accent_bgr, 1)
            cv2.putText(canvas, f"EMO: {data['emotion'].upper()}", (10, 30), font, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas, f"STATUS: {data['focus']}", (10, 45), font, 0.3, (0, 255, 150), 1)

            # Save Image
            cv2.imwrite(img_path, canvas)
            
            # --- DATABASE ARCHIVING ---
            db_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'identity': data.get('identity', 'Unknown'),
                'gender': data['gender'],
                'age': data['age'],
                'emotion': data['emotion'],
                'focus_score': data.get('yaw', 0.0), # Using yaw as a proxy for focus score for now
                'gesture': data.get('gesture', 'None'),
                'image_path': img_path
            }
            db_manager.add_record(db_record)
            
            # Save Metadata (Backward Compatibility)
            meta_path = os.path.join(folder, f"{filename}.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"NEXUS BIOMETRIC REPORT\n")
                f.write(f"======================\n")
                f.write(f"TIMESTAMP: {db_record['timestamp']}\n")
                f.write(f"IDENTITY : {db_record['identity']}\n")
                f.write(f"GENDER   : {data['gender']}\n")
                f.write(f"AGE      : {data['age']} years (Conf: {data['age_conf']:.2f})\n")
                f.write(f"EMOTION  : {data['emotion']}\n")
                f.write(f"FOCUS    : {data['focus']}\n")
                f.write(f"POSE     : Yaw={data['yaw']:.1f}, Pitch={data['pitch']:.1f}\n")
                f.write(f"LOCATION : {data['bbox']}\n")
            
            self.stream.add_log(f"ARCHIVED TO DB: {filename}")
        except Exception as e:
            print(f"ARCHIVE ERROR: {e}")

    def init_ui(self):
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.layout = QHBoxLayout(self.central)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # --- LEFT SIDECAR ---
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(280)
        self.side_layout = QVBoxLayout(self.sidebar)
        self.side_layout.setContentsMargins(20, 20, 20, 20)
        
        # Pulse Logo
        self.logo_container = QWidget()
        self.logo_container.setFixedHeight(100)
        logo_lay = QVBoxLayout(self.logo_container)
        self.logo = QLabel("NEXUS OS")
        self.logo.setFont(QFont("Impact", 32))
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_lay.addWidget(self.logo)
        
        self.status_box = QFrame()
        status_lay = QVBoxLayout(self.status_box)
        self.lbl_status = QLabel("BOOTING...")
        self.lbl_info = QLabel("KERNEL: 60FPS\nAI: ASYNC")
        self.lbl_info.setStyleSheet("font-size: 9px; color: #888;")
        status_lay.addWidget(self.lbl_status)
        status_lay.addWidget(self.lbl_info)
        
        # Analytics
        ana_title = QLabel("EMOTION SPECTRUM")
        ana_title.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")
        self.chart = AnalyticsChart()
        
        # Data Stream
        stream_title = QLabel("LIVE DATA STREAM")
        stream_title.setStyleSheet("color: #888; font-size: 10px; font-weight: bold; margin-top: 10px;")
        self.stream = DataStreamWidget()

        # Neural Wave
        wave_title = QLabel("NEURAL FEEDBACK")
        wave_title.setStyleSheet("color: #888; font-size: 10px; font-weight: bold; margin-top: 10px;")
        self.wave = NeuralWaveform()
        
        self.btn_theme = QPushButton("☀️ DAY MODE")
        self.btn_theme.setFixedHeight(60)
        self.btn_theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme.clicked.connect(self.toggle_theme)
        
        self.btn_vision = QPushButton("👁️ CYCLE VISION")
        self.btn_vision.setFixedHeight(40)
        self.btn_vision.setStyleSheet("background: #333; color: #fff; border-radius: 10px;")
        self.btn_vision.clicked.connect(self.cycle_vision)
        
        self.btn_exit = QPushButton("TERMINATE")
        self.btn_exit.setFixedHeight(50)
        self.btn_exit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_exit.clicked.connect(self.close)
        
        self.side_layout.addWidget(self.logo_container)
        self.side_layout.addSpacing(10)
        self.side_layout.addWidget(self.status_box)
        self.side_layout.addSpacing(15)
        self.side_layout.addWidget(ana_title)
        self.side_layout.addWidget(self.chart)
        self.side_layout.addWidget(stream_title)
        self.side_layout.addWidget(self.stream)
        self.side_layout.addWidget(wave_title)
        self.side_layout.addWidget(self.wave)
        self.side_layout.addStretch()
        self.side_layout.addWidget(self.btn_vision)
        self.side_layout.addSpacing(10)
        self.side_layout.addWidget(self.btn_theme)
        self.side_layout.addSpacing(10)
        self.side_layout.addWidget(self.btn_exit)
        
        # --- CENTER FEED ---
        self.display = VideoWidget()
        
        # --- RIGHT TAB SYSTEM ---
        self.right_tabs = QTabWidget()
        self.right_tabs.setFixedWidth(350)
        
        # Tab 1: Live Feed
        self.gallery_side = QFrame()
        gallery_outer = QVBoxLayout(self.gallery_side)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.gallery_content = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_content)
        self.gallery_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.gallery_content)
        gallery_outer.addWidget(self.scroll)
        self.right_tabs.addTab(self.gallery_side, "LIVE FEED")
        
        # Tab 2: History
        self.history_side = QFrame()
        history_lay = QVBoxLayout(self.history_side)
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["Time", "ID", "Age", "Emotion"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setStyleSheet("background: #111; color: #00f2ff; gridline-color: #333;")
        history_lay.addWidget(self.history_table)
        self.btn_refresh = QPushButton("REFRESH RECORDS")
        self.btn_refresh.clicked.connect(self.load_history)
        history_lay.addWidget(self.btn_refresh)
        self.right_tabs.addTab(self.history_side, "HISTORY")
        
        # Tab 3: Settings
        self.settings_side = QFrame()
        settings_lay = QVBoxLayout(self.settings_side)
        settings_lay.addWidget(QLabel("AI SENSITIVITY"))
        settings_lay.addWidget(QLabel("VOICE CONTROL: ENABLED"))
        settings_lay.addWidget(QLabel("GESTURE CONTROL: ENABLED"))
        settings_lay.addStretch()
        self.right_tabs.addTab(self.settings_side, "SETTINGS")
        
        self.layout.addWidget(self.sidebar)
        self.layout.addWidget(self.display)
        self.layout.addWidget(self.right_tabs)
        
        self.apply_theme()
        self.load_history()

    def load_history(self):
        try:
            conn = sqlite3.connect(db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, identity, age, emotion FROM records ORDER BY id DESC LIMIT 50")
            rows = cursor.fetchall()
            self.history_table.setRowCount(0)
            for row_idx, row_data in enumerate(rows):
                self.history_table.insertRow(row_idx)
                for col_idx, value in enumerate(row_data):
                    self.history_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
            conn.close()
        except: pass

    def toggle_theme(self):
        shared_state.theme_mode = "Day" if shared_state.theme_mode == "Night" else "Night"
        self.btn_theme.setText("🌙 SWITCH TO NIGHT" if shared_state.theme_mode == "Day" else "☀️ SWITCH TO DAY")
        self.apply_theme()

    def cycle_vision(self):
        modes = ["Normal", "Matrix", "Thermal"]
        curr = shared_state.vision_mode
        idx = (modes.index(curr) + 1) % len(modes)
        shared_state.vision_mode = modes[idx]
        self.lbl_status.setText(f"VISION: {shared_state.vision_mode.upper()}")
        
        if HAS_TTS:
            try:
                engine = pyttsx3.init()
                engine.say(f"Vision switched to {shared_state.vision_mode}")
                engine.runAndWait()
            except: pass

    def handle_voice_command(self, cmd):
        if cmd == "SNAPSHOT":
            self.update_gallery()
            self.stream.add_log("VOICE: SNAPSHOT TRIGGERED")
        elif cmd == "TOGGLE_THEME":
            self.toggle_theme()
            self.stream.add_log("VOICE: THEME TOGGLED")
        elif cmd == "CYCLE_VISION":
            self.cycle_vision()
            self.stream.add_log("VOICE: VISION CYCLED")
        elif cmd == "EXIT":
            self.close()

    def handle_gesture(self, action):
        if action == "SNAPSHOT":
            self.update_gallery()
            self.stream.add_log("GESTURE: SNAPSHOT TRIGGERED")
        elif action == "RESET_THREAT":
            shared_state.mutex.lock()
            shared_state.threat_level = 0.0
            shared_state.mutex.unlock()
            self.stream.add_log("GESTURE: THREAT RESET")
        elif action == "CYCLE_VISION":
            # Avoid cycling too fast
            now = time.time()
            if not hasattr(self, 'last_cycle_time') or now - self.last_cycle_time > 1.5:
                self.cycle_vision()
                self.last_cycle_time = now
                self.stream.add_log("GESTURE: VISION CYCLED")

    def handle_greeting(self, emo, age, gender):
        # AI Logic to choose message
        if gender == "SYSTEM":
            msg = "Warning! Hostile intent detected. Please remain calm. Security protocols active."
        elif "Happy" in emo:
            msg = f"Hello! I detect a happy {gender} who looks about {age} years old. Your smile is wonderful!"
        elif "Angry" in emo:
            msg = f"I notice you are angry. Is everything alright? You are a strong {gender} of {age} years."
        elif "Sad" in emo:
            msg = f"I sense sadness. Don't worry, things will get better. You look like a kind {gender}."
        elif "Surprise" in emo:
            msg = f"Wow! You look very surprised! My biometric scan of a {age} year old {gender} is peaking."
        else:
            msg = f"Biometric link stable. Analyzing a {age} year old {gender}."

        self.lbl_status.setText("NEXUS: ANALYZING...")
        if HAS_TTS:
            try:
                class VoiceTask(QThread):
                    def run(self):
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.say(msg)
                        engine.runAndWait()
                self.v_task = VoiceTask()
                self.v_task.start()
            except: pass

    def apply_theme(self):
        if shared_state.theme_mode == "Night":
            bg = "#08080c"
            side = "#0e0e14"
            text = "#00f2ff"
            accent = "#00f2ff"
        else:
            bg = "#ffffff"
            side = "#f5f5f5"
            text = "#c5a000" # GOLD
            accent = "#c5a000"

        self.sidebar.setStyleSheet(f"background-color: {side}; border-right: 1px solid {accent}44;")
        self.gallery_side.setStyleSheet(f"background-color: {bg}; border-left: 1px solid {accent}44;")
        self.logo.setStyleSheet(f"color: {text}; letter-spacing: 5px;")
        self.status_box.setStyleSheet(f"border: 1px solid {accent}88; border-radius: 15px; background: rgba(0,0,0,0.05);")
        self.lbl_status.setStyleSheet(f"color: {accent}; font-weight: bold; font-size: 14px;")
        
        self.btn_theme.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {text};
                border: 2px solid {accent};
                border-radius: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {accent};
                color: white;
            }}
        """)
        
        self.btn_exit.setStyleSheet("""
            QPushButton { background: #ff0055; color: white; border-radius: 10px; border: none; }
            QPushButton:hover { background: #ff3377; }
        """)

    def closeEvent(self, event):
        shared_state.is_running = False
        self.cam_thread.wait()
        self.ai_thread.wait()
        if hasattr(self, 'voice_thread'):
            self.voice_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NexusLuxuryApp()
    window.show()
    sys.exit(app.exec())