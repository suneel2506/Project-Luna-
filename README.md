# 🤖 Interactive AI Vision Assistant

An advanced AI-powered system that combines **Computer Vision + Machine Learning + Human-in-the-loop Learning** to create an intelligent assistant that can see, understand, and learn from users in real time.

---

## 🚀 Features

### 👁️ Computer Vision Capabilities

* 📦 **Object Detection** (YOLO-based)
* 👤 **Face Recognition**

  * Detects known faces
  * Asks user to label unknown faces
  * Learns and remembers identities
* 😊 **Emotion Detection**

  * Detects emotions like Happy, Sad, Angry, Neutral
* ✋ **Sign Language Detection**

  * Recognizes basic gestures (Hello, Yes, No, Thank You, etc.)

---

### 🧠 Intelligent Learning System

* Human-in-the-loop learning approach
* When unknown object/face is detected:

  * System asks user for input
  * Stores new knowledge
  * Improves over time

---

### 🌍 Multi-language Support

* Outputs translated into:

  * English
  * Tamil
  * Hindi

---

### 💬 Modern Chat Interface

* Chat-based interaction (like ChatGPT/WhatsApp)
* Real-time AI responses
* Clean UI built with React + Tailwind CSS

---

## 🏗️ Tech Stack

### 🔹 Backend

* Python
* Flask
* OpenCV
* MediaPipe
* face_recognition
* Ultralytics YOLO

### 🔹 Frontend

* React (Vite)
* Tailwind CSS
* Axios

### 🔹 Storage

* JSON / SQLite (for learned data)

---

## 📁 Project Structure

```
AI_Vision_Assistant/
│
├── backend/
│   ├── app.py
│   ├── models/
│   ├── learning/
│   └── database/
│
├── frontend/
│   ├── src/
│   ├── components/
│   └── styles/
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/suneel2506/ai-vision-assistant.git
cd ai-vision-assistant
```

---

### 🔹 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

---

### 🔹 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## ▶️ Usage

1. Open the frontend in your browser
2. Start the AI assistant
3. Allow camera access
4. Interact via chat:

   * AI detects objects, faces, emotions, gestures
   * If unknown → asks for input
   * Learns and improves

---

## 🧠 How It Works

1. User provides image/video input
2. Backend processes using AI models
3. Results sent to frontend
4. Displayed in chat UI
5. Unknown data → user labels → stored for future use

---

## 💡 Future Enhancements

* 🎥 Live video streaming optimization
* 🗣️ Voice assistant integration
* ☁️ Cloud deployment
* 📱 Mobile app version
* 🧠 Advanced deep learning models

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork the repo and improve the system.

---

## 📜 License

This project is for educational purposes.

---

## 🙌 Acknowledgment

Built as an AIML project to demonstrate real-world AI integration with interactive learning systems.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
