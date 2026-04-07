# 🌙 Project LUNA — Interactive AI Vision Assistant

An AI-powered vision assistant that **sees**, **understands**, **talks**, and **learns** — with a stunning modern dark-themed chat UI.

---

## 🚀 Features

### 👁️ Computer Vision
- 📦 **Object Detection** — YOLOv8 nano model for real-time detection
- 👤 **Face Recognition** — Encoding-based face matching with learning
- 😊 **Emotion Detection** — DeepFace-powered emotion classification
- 🤟 **Sign Language Detection** — MediaPipe hand landmark analysis (Hello, Yes, No, Thank You, Help, I Love You)

### 🧠 Human-in-the-Loop Learning
- Unknown faces trigger a learning prompt — user labels them, LUNA remembers
- Unknown objects can be taught — persistent JSON storage
- Knowledge persists across sessions

### 🌍 Multi-Language Support
- English 🇬🇧
- Tamil (தமிழ்) 🇮🇳
- Hindi (हिंदी) 🇮🇳

### 💬 Modern Chat Interface
- Dark glassmorphic design with purple/cyan gradients
- Real-time AI responses with detection badges
- Camera preview with live detection overlay
- Responsive — works on desktop and mobile

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)                                 │
│  ┌──────────┐ ┌───────────────┐ ┌──────────────────────┐ │
│  │ Sidebar  │ │ Camera Panel  │ │ Chat Panel           │ │
│  │ • Nav    │ │ • Webcam Feed │ │ • Message Bubbles    │ │
│  │ • Toggles│ │ • Detection   │ │ • Detection Chips    │ │
│  │ • Stats  │ │   Overlays    │ │ • Learning Prompts   │ │
│  └──────────┘ └───────┬───────┘ └──────────────────────┘ │
└───────────────────────┼──────────────────────────────────┘
                        │ Axios HTTP (base64 frames)
┌───────────────────────▼──────────────────────────────────┐
│  Backend (Flask API)                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │ YOLO       │ │ Face Rec   │ │ Emotion    │           │
│  │ Detection  │ │ Module     │ │ Detection  │           │
│  ├────────────┤ ├────────────┤ ├────────────┤           │
│  │ Sign       │ │ Response   │ │ Translator │           │
│  │ Detection  │ │ Generator  │ │ (EN/TA/HI) │           │
│  └────────────┘ └────────────┘ └────────────┘           │
│                  ┌────────────┐                          │
│                  │ Learning   │ ← Human-in-the-loop     │
│                  │ Handler    │                          │
│                  └─────┬──────┘                          │
└────────────────────────┼─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Storage                                                 │
│  learned_objects.json  •  learned_faces.json  •  faces/  │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Project LUNA/
├── backend/
│   ├── app.py                         # Flask API server
│   ├── config.py                      # Configuration settings
│   ├── requirements.txt               # Python dependencies
│   │
│   ├── models/                        # AI detection modules
│   │   ├── object_detection.py        # YOLO-based detection
│   │   ├── face_recognition_module.py # Face encoding & matching
│   │   ├── emotion_detection.py       # Emotion classification
│   │   └── sign_detection.py          # MediaPipe sign detection
│   │
│   ├── learning/                      # Human-in-the-loop learning
│   │   └── unknown_handler.py         # Manage unknown detections
│   │
│   ├── utils/                         # Utilities
│   │   ├── translator.py              # Multi-language translation
│   │   └── response_generator.py      # Natural language responses
│   │
│   ├── storage/                       # Persistent data
│   │   ├── learned_objects.json
│   │   ├── learned_faces.json
│   │   └── faces/                     # Face encoding files
│   │
│   └── tests/
│       └── test_api.py                # API tests
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   │
│   ├── public/
│   │   └── luna-icon.svg              # Moon-themed favicon
│   │
│   └── src/
│       ├── main.jsx                   # Entry point
│       ├── App.jsx                    # Main app orchestrator
│       ├── index.css                  # Design system & styles
│       │
│       ├── components/
│       │   ├── Header.jsx             # Status + language selector
│       │   ├── Sidebar.jsx            # Navigation + detector toggles
│       │   ├── CameraPreview.jsx      # Webcam feed + overlays
│       │   ├── ChatWindow.jsx         # Message list + typing indicator
│       │   ├── ChatMessage.jsx        # Message bubbles + chips
│       │   ├── ChatInput.jsx          # Text input + detect toggle
│       │   └── LearningModal.jsx      # Label unknown detections
│       │
│       ├── hooks/
│       │   ├── useCamera.js           # Camera access hook
│       │   └── useChat.js             # Chat state management
│       │
│       ├── services/
│       │   └── api.js                 # Axios API client
│       │
│       └── utils/
│           └── constants.js           # App constants
│
└── README.md
```

---

## ⚙️ Prerequisites

- **Python** 3.9+
- **Node.js** 18+
- **CMake** (optional, for `face_recognition` library)
- **Webcam** (for camera features)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/suneel2506/Project-Luna-.git
cd "Project LUNA"
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

The backend will start at `http://localhost:5000`.

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The frontend will start at `http://localhost:5173`.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Health check — module status, learned items count |
| `POST` | `/api/process` | Process a camera frame through all AI detectors |
| `POST` | `/api/learn` | Submit user label for unknown face/object |
| `GET` | `/api/history` | Get all learned items summary |
| `GET` | `/api/languages` | Get supported languages |

### POST `/api/process`

**Request:**
```json
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
```

**Response:**
```json
{
  "objects": [{"label": "bottle", "confidence": 0.92, "bbox": [x,y,w,h]}],
  "faces": [{"name": "Suneel", "location": [t,r,b,l]}],
  "emotions": [{"emotion": "happy", "confidence": 0.85}],
  "signs": [{"sign": "hello", "confidence": 0.9}],
  "unknown_faces": [{"id": "face_001", "crop_base64": "..."}],
  "message": "I see Suneel looking happy! There's a bottle on the table.",
  "translations": {
    "en": "...",
    "ta": "...",
    "hi": "..."
  }
}
```

### POST `/api/learn`

**Request:**
```json
{
  "type": "face",
  "id": "face_001",
  "label": "Suneel"
}
```

---

## ▶️ Usage

1. Open `http://localhost:5173` in your browser
2. Click **▶ Start** to activate the camera
3. Click **🔍 Detect** to begin AI analysis (auto-captures every 2s)
4. LUNA will respond in the chat with what it sees
5. If LUNA encounters unknowns, it asks you to label them
6. Switch languages with the EN/தமிழ்/हிंदி buttons
7. Toggle individual detectors on/off in the sidebar

---

## 🧪 Running Tests

```bash
cd backend
python tests/test_api.py
```

---

## 💡 Future Enhancements

- 🎥 WebSocket streaming for faster detection
- 🗣️ Voice assistant integration (text-to-speech)
- ☁️ Cloud deployment (Docker + AWS/GCP)
- 📱 Mobile app (React Native / Capacitor)
- 🧠 Fine-tuned custom detection models

---

## 🤝 Contributing

Contributions are welcome! Fork the repo and submit a pull request.

---

## 📜 License

This project is for educational purposes.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
