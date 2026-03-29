# 🦅 NEXUS // BIOMETRIC MASTER 💎
> **"الذكاء الاصطناعي الذي يرى ما وراء الملامح"** | **"AI that sees beyond the features"**

---

## 📥 Data Download | تحميل البيانات
To run the project, you need the trained models and the dataset. You can download them from the following link:
لإكمال تشغيل المشروع، ستحتاج إلى النماذج المدربة والبيانات. يمكنك تحميلها من الرابط التالي:

> [**Download Dataset | تحميل البيانات**](https://drive.google.com/file/d/1X2hQ8ovxYwCv3Lz1l5Mt_zCzKSJHnYqp/view?usp=drive_link)

---

## 📖 نظرة شاملة على مشروع NEXUS
مشروع **NEXUS** هو نظام استخباراتي بيومتري متكامل يجمع بين قوة الرؤية الحاسوبية (Computer Vision) وجمالية واجهات الخيال العلمي. يهدف النظام إلى تقديم تحليل لحظي ودقيق للحالة الإنسانية أمام الكاميرا.

### 🛠️ البنية التحتية والتقنيات المستخدمة
يعتمد النظام على بنية متطورة تضمن الأداء العالي:
1. **المعالجة غير المتزامنة (Asynchronous Multi-Threading)**:
   - يتم فصل العمليات الحسابية الثقيلة للذكاء الاصطناعي في خيوط (Threads) مستقلة عن واجهة المستخدم.
   - يضمن هذا بقاء الواجهة سلسة (Smooth UI) حتى أثناء تشغيل أقوى النماذج.
2. **محرك الذكاء الاصطناعي الهجين**:
   - **التعرف على المشاعر**: يستخدم نموذج `EfficientNet-B4` الذي يتميز بالقدرة على فهم التفاصيل الدقيقة للملامح وتجاوز مشكلة الإضاءة المتغيرة.
   - **تحليل العمر والجنس**: يعتمد على نماذج `VGG-16` مخصصة تم تحسينها لتعمل بسرعة عالية وبأقل استهلاك للذاكرة (Memory Optimization).
3. **خوارزميات تتبع التركيز والوضعية**:
   - يستخدم النظام معادلات هندسية لحساب زوايا الرأس (Yaw, Pitch, Roll) لتحديد ما إذا كان الشخص في حالة تركيز كامل أو تشتت.

### 🌟 المميزات والمزايا الإضافية
- **التصويت الزمني (Temporal Smoothing)**: خوارزمية ذكية تقوم بتحليل سلسلة من الإطارات المتتابعة لاتخاذ القرار النهائي بشأن الشعور، مما يمنع "الارتعاش" في النتائج (Stabilized Predictions).
- **نظام الأرشفة الأمني**: يتم حفظ كل لقطة بترميز HUD رقمي (Heads-Up Display) يحتوي على التاريخ، الوقت، ونوع الشعور، مع تخزين كافة البيانات في قاعدة بيانات SQLite محميّة.
- **تفاعل صوتي ذكي**: يدعم النظام تحويل النص إلى كلام (TTS) للترحيب بالمستخدمين وتقديم تنبيهات صوتية عند رصد تغييرات معينة.

---

<a name="english"></a>
## 📖 Comprehensive Project Overview
**NEXUS** is an integrated biometric intelligence suite that merges high-performance computer vision with a futuristic Sci-Fi aesthetic. It provides precise, real-time insights into the human state.

### 🛠️ Infrastructure & Technologies
The system is built on a robust architecture to ensure peak performance:
1. **Asynchronous Multi-Threading**:
   - Heavy AI computational tasks are decoupled from the main UI thread using independent QThreads.
   - This ensures a "lag-free" user experience even while running multiple deep learning models simultaneously.
2. **Hybrid AI Engine**:
   - **Emotion Recognition**: Powered by `EfficientNet-B4`, which captures intricate facial details and handles varying lighting conditions effectively.
   - **Age & Gender Estimation**: Utilizes optimized `VGG-16` architectures tailored for high-speed inference and minimal memory footprint.
3. **Focus & Pose Tracking**:
   - Employs geometric algorithms to calculate head pose angles (Yaw, Pitch, Roll), accurately determining user engagement and attention levels.

### 🌟 Core Advantages & Features
- **Temporal Voting (Prediction Stabilization)**: A proprietary algorithm that analyzes recent frames to cast a final vote on the dominant emotion, eliminating jitter and flickering.
- **Advanced HUD Archiving**: Every snapshot is saved with a digital overlay (HUD) containing tech metadata (timestamp, results) and logged into an encrypted SQLite database.
- **Intelligent Voice Feedback**: Integrated Text-to-Speech (TTS) capabilities for interactive greetings and real-time status alerts.

---

## 📂 Project Structure | هيكلية الملفات
```text
Emotion-Recognition-System/
├── combined_app.py        # The Master Hub & Main UI
├── app.py                 # Core GUI Logic (PyQt6)
├── age_app.py             # Focused Age Detection Logic
├── gander_app.py          # Focused Gender Detection Logic
├── train_model.py         # Local Training Script
├── requirements.txt       # Essential Dependencies
├── nexus_biometrics.db    # Professional SQLite Database
└── models/                # AI Model Weights & Metadata
```

# ⚙️ Installation & Usage | التثبيت والتشغيل

### 1. Requirements | المتطلبات
- Python 3.8+ | 8GB RAM Minimum
- Webcam | Optional: NVIDIA GPU (CUDA)

### 2. Setup | الإعداد
```bash
# Install all dependencies | تثبيت كافة المكتبات
pip install torch torchvision opencv-python Pillow tensorflow tf_keras pyttsx3 PyQt6
```

### 3. Execution | البدء
```bash
# Run the combined intelligence center | تشغيل مركز الاستخبارات الرئيسي
python combined_app.py
```

---

## 🛡️ License & Credit
Developed with passion by **ENG.Bilal_Al-Shami** 🦅💎

