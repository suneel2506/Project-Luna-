# 🌙 Project LUNA — Interactive AI Vision Assistant

An intelligent vision assistant that **sees**, **understands**, **talks**, and **learns** from you.

LUNA uses real-time computer vision to detect objects, recognize faces, read emotions, and interpret sign language — all through a conversational chat interface with multi-language support.

---

## ✨ Features

| Feature | Technology | Status |
|---|---|---|
| **Object Detection** | YOLOv8 (Ultralytics) | ✅ Real-time |
| **Face Recognition** | face_recognition + OpenCV | ✅ Learning-capable |
| **Emotion Detection** | DeepFace | ✅ 7 emotions |
| **Sign Language** | MediaPipe Hand Tracking | ✅ 6 gestures |
| **Multi-Language** | Local dictionaries + googletrans | ✅ EN / TA / HI |
| **Human-in-the-Loop Learning** | Custom learning system | ✅ Faces + Objects |
| **Chat Interface** | React + Tailwind CSS | ✅ WhatsApp-style |

---

## 🧱 Tech Stack

### Backend (Python 3.11+)
- **Flask** — REST API server
- **OpenCV** — Image processing & face detection
- **Ultralytics YOLOv8** — Object detection
- **MediaPipe** — Hand landmark detection
- **face_recognition** — Face encoding & matching (optional)
- **DeepFace** — Emotion analysis (optional)

### Frontend (React + Vite)
- **React 18** — UI framework
- **Tailwind CSS** — Styling
- **Axios** — API communication

---

## 📁 Project Structure

```
Project LUNA/
├── backend/
│   ├── app.py                          # Flask app (factory pattern)
│   ├── config.py                       # Centralized configuration
│   ├── requirements.txt                # Python dependencies
│   ├── models/
│   │   ├── object_detection.py         # YOLOv8 detector
│   │   ├── face_recognition_module.py  # Face detect + encode + match
│   │   ├── emotion_detection.py        # DeepFace emotions
│   │   └── sign_detection.py           # MediaPipe hand gestures
│   ├── learning/
│   │   └── unknown_handler.py          # Human-in-the-loop learning
│   ├── utils/
│   │   ├── translator.py              # Multi-language translation
│   │   └── response_generator.py      # Natural language responses
│   ├── storage/
│   │   ├── faces/                     # Face image crops
│   │   ├── learned_faces.json         # Known face encodings
│   │   └── learned_objects.json       # Learned object labels
│   └── tests/
│       ├── conftest.py                # Shared test fixtures
│       └── test_api.py               # API test suite (24 tests)
└── frontend/
    └── src/                           # React application
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11.9+ (recommended)
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create & activate virtual environment
python -m venv ../luna_env
# Windows:
..\luna_env\Scripts\activate
# macOS/Linux:
source ../luna_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install enhanced AI features
pip install face_recognition    # Requires dlib + CMake
pip install deepface            # Emotion detection
pip install googletrans==4.0.0-rc1  # Dynamic translation

# Run the server
python app.py
```

The backend runs at **http://localhost:5000**.

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The frontend runs at **http://localhost:5173**.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Health check + module status |
| `POST` | `/api/process` | Process camera frame through all detectors |
| `POST` | `/api/learn` | Submit user label for unknown detection |
| `GET` | `/api/history` | Get learned items summary |
| `GET` | `/api/languages` | List supported languages |

### POST /api/process

```json
// Request
{
  "image": "base64_encoded_image",
  "language": "en",
  "detectors": {
    "objects": true,
    "faces": true,
    "emotions": true,
    "signs": true
  }
}

// Response
{
  "objects": [{"label": "laptop", "confidence": 0.92, "bbox": [x1,y1,x2,y2]}],
  "faces": [{"name": "Unknown", "location": [...], "is_unknown": true, "face_id": "face_abc123"}],
  "emotions": [{"emotion": "happy", "confidence": 0.85}],
  "signs": [{"sign": "hello", "confidence": 0.85, "hand": "right"}],
  "message": "I can see a laptop. I see someone I don't recognize. Who is this?",
  "translations": {"en": "...", "ta": "...", "hi": "..."}
}
```

### POST /api/learn

```json
// Request
{"type": "face", "id": "face_abc123", "label": "John"}

// Response
{"success": true, "message": "Got it! I'll remember John from now on. 🧠✨"}
```

---

## 🧪 Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

**24 tests** covering all endpoints with valid/invalid inputs.

---

## 📋 Architecture Highlights

- **App Factory Pattern** — `create_app()` for clean initialization & testing
- **Thread-safe AI inference** — Mutex locks on YOLO model and face encodings
- **Frame-rate throttling** — Server-side rate limiting to prevent overload
- **Message deduplication** — Suppresses identical messages within time window
- **Graceful degradation** — Every AI module falls back cleanly when its library is missing
- **IoU deduplication** — Removes overlapping object detection boxes
- **Gesture debouncing** — Prevents rapid-fire repeated sign detections
- **Automatic stale cleanup** — Pending unknowns expire after configurable TTL

---

## 🌐 Supported Languages

| Code | Language | Coverage |
|---|---|---|
| `en` | English | Full |
| `ta` | Tamil | 60+ words/phrases |
| `hi` | Hindi | 60+ words/phrases |

---

## 🤟 Supported Sign Language Gestures

| Gesture | Sign |
|---|---|
| ✋ Open palm | Hello |
| 👍 Thumbs up | Yes |
| ✌️ Peace sign | No |
| 🙏 Flat hand | Thank You |
| ✊ Closed fist | Help |
| 🤟 Rock on | I Love You |

---

## 📄 License

This project is for educational purposes.
