/**
 * Project LUNA — Main App Component
 * Orchestrates sidebar, camera, chat, and learning flow.
 */

import { useState, useCallback, useEffect } from 'react';
import './index.css';

import Header from './components/Header';
import Sidebar from './components/Sidebar';
import CameraPreview from './components/CameraPreview';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import LearningModal from './components/LearningModal';

import { useCamera } from './hooks/useCamera';
import { useChat } from './hooks/useChat';
import { processFrame, submitLabel, getStatus, getHistory } from './services/api';

function App() {
  // ---- State ----
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeView, setActiveView] = useState('chat');
  const [language, setLanguage] = useState('en');
  const [backendStatus, setBackendStatus] = useState('offline');
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectors, setDetectors] = useState({
    objects: true,
    faces: true,
    emotions: true,
    signs: true,
  });
  const [latestResults, setLatestResults] = useState(null);
  const [learningItem, setLearningItem] = useState(null);
  const [learnedStats, setLearnedStats] = useState({ faces: 0, objects: 0 });

  // ---- Hooks ----
  const camera = useCamera();
  const chat = useChat();

  // ---- Check backend status on mount ----
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await getStatus();
        setBackendStatus(status.status || 'online');
        if (status.learned) {
          setLearnedStats({
            faces: status.learned.faces_count || 0,
            objects: status.learned.objects_count || 0,
          });
        }
      } catch {
        setBackendStatus('offline');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // ---- Process a captured frame ----
  const handleCapture = useCallback(async () => {
    if (chat.isProcessing) return;

    const frameData = camera.captureFrame();
    if (!frameData) {
      // Frame not ready yet — this is normal during camera warmup
      return;
    }

    chat.setIsProcessing(true);

    try {
      const results = await processFrame(frameData, language, detectors);

      // Backend may throttle — skip silently
      if (results.skipped) {
        return;
      }

      // Check if there are any real detections
      const hasDetections =
        (results.objects?.length > 0) ||
        (results.faces?.length > 0) ||
        (results.emotions?.length > 0) ||
        (results.signs?.length > 0);

      // Always update overlay when real detections exist
      if (hasDetections) {
        setLatestResults(results);
        // Add chat message — show even "no detection" messages
        if (results.message) {
          chat.addAIMessage(results.message, results);
        }
      } else {
        setLatestResults(null);
        // Still show the message if the backend generated one (e.g. "nothing detected")
        if (results.message) {
          chat.addAIMessage(results.message);
        }
      }

      // Check for unknown faces → trigger learning
      if (results.unknown_faces?.length > 0) {
        const unknownFace = results.unknown_faces[0];
        setLearningItem({
          type: 'face',
          id: unknownFace.face_id,
          crop_base64: unknownFace.crop_base64 || '',
        });
        chat.addLearningMessage(
          "I see a face I don't recognize! Can you tell me who this is?"
        );
      }
    } catch (err) {
      console.error('Frame processing error:', err);
      // Don't spam errors for throttled responses (429)
      if (err?.response?.status === 429) {
        return;
      }
      if (err.code === 'ERR_NETWORK') {
        chat.addSystemMessage('⚠️ Cannot reach LUNA backend. Is it running?');
        setBackendStatus('offline');
      } else {
        chat.addSystemMessage(`⚠️ Processing error: ${err.message}`);
      }
    } finally {
      chat.setIsProcessing(false);
    }
  }, [camera, chat, language, detectors]);

  // ---- Handle user text messages ----
  const handleSendMessage = useCallback(
    (text) => {
      chat.addUserMessage(text);

      // Simple responses for text commands
      const lower = text.toLowerCase();
      if (lower.includes('clear') || lower.includes('reset')) {
        chat.clearMessages();
      } else if (lower.includes('hello') || lower.includes('hi')) {
        setTimeout(() => {
          chat.addAIMessage(
            "Hello! 🌙 I'm LUNA, your AI Vision Assistant. Start your camera and click Detect to see me in action!"
          );
        }, 500);
      } else if (lower.includes('help')) {
        setTimeout(() => {
          chat.addAIMessage(
            "Here's what I can do:\n📦 Detect objects (YOLO)\n👤 Recognize faces\n😊 Read emotions\n🤟 Understand sign language\n🧠 Learn new faces and objects from you!\n\nStart your camera and click 🔍 Detect to begin."
          );
        }, 500);
      } else {
        setTimeout(() => {
          chat.addAIMessage(
            "I understand text, but I'm best at vision! Start your camera and click 🔍 Detect to see my full capabilities."
          );
        }, 500);
      }
    },
    [chat]
  );

  // ---- Toggle detection ----
  const handleToggleDetect = useCallback(() => {
    const next = !isDetecting;
    setIsDetecting(next);
    if (next) {
      chat.addSystemMessage('🔍 Detection started — analyzing frames every 2 seconds');
    } else {
      chat.addSystemMessage('⏹ Detection stopped');
      setLatestResults(null);
    }
  }, [isDetecting, chat]);

  // ---- Toggle individual detector ----
  const handleToggleDetector = useCallback((key) => {
    setDetectors((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  // ---- Handle learning submission ----
  const handleLearnSubmit = useCallback(
    async (type, id, label) => {
      try {
        const result = await submitLabel(type, id, label);
        if (result.success) {
          chat.addSystemMessage(
            `✅ ${result.message || `Learned "${label}" successfully!`}`
          );
          // Update stats
          if (result.learned_summary) {
            setLearnedStats({
              faces: result.learned_summary.faces_count || learnedStats.faces,
              objects: result.learned_summary.objects_count || learnedStats.objects,
            });
          }
        } else {
          chat.addSystemMessage(`❌ Couldn't learn "${label}". Try again.`);
        }
      } catch (err) {
        chat.addSystemMessage(`⚠️ Learning error: ${err.message}`);
      }
      setLearningItem(null);
    },
    [chat, learnedStats]
  );

  // ---- Learned items view ----
  const [learnedItems, setLearnedItems] = useState(null);
  useEffect(() => {
    if (activeView === 'learned') {
      getHistory()
        .then(setLearnedItems)
        .catch(() => setLearnedItems(null));
    }
  }, [activeView]);

  // ---- Camera start/stop with chat feedback ----
  const handleStartCamera = useCallback(async () => {
    const success = await camera.startCamera();
    if (success) {
      chat.addSystemMessage('📷 Camera started! Click 🔍 Detect to begin analyzing.');
    }
  }, [camera, chat]);

  const handleStopCamera = useCallback(() => {
    camera.stopCamera();
    setIsDetecting(false);
    setLatestResults(null);
    chat.addSystemMessage('📷 Camera stopped.');
  }, [camera, chat]);

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        activeView={activeView}
        onViewChange={setActiveView}
        detectors={detectors}
        onToggleDetector={handleToggleDetector}
        learnedStats={learnedStats}
      />

      {/* Main Content */}
      <div className="main-content">
        <Header
          language={language}
          onLanguageChange={setLanguage}
          backendStatus={backendStatus}
          isProcessing={chat.isProcessing}
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
        />

        <div className="content-area">
          {/* Camera Panel */}
          <CameraPreview
            videoRef={camera.videoRef}
            canvasRef={camera.canvasRef}
            isCameraOn={camera.isCameraOn}
            cameraError={camera.cameraError}
            onStartCamera={handleStartCamera}
            onStopCamera={handleStopCamera}
            onCapture={handleCapture}
            isDetecting={isDetecting}
            isProcessing={chat.isProcessing}
            detectionResults={latestResults}
          />

          {/* Right Panel: Chat or Learned */}
          <div className="chat-panel">
            {activeView === 'chat' || activeView === 'settings' ? (
              <>
                <ChatWindow
                  messages={chat.messages}
                  scrollRef={chat.scrollRef}
                  isProcessing={chat.isProcessing}
                />
                <ChatInput
                  onSendMessage={handleSendMessage}
                  isDetecting={isDetecting}
                  onToggleDetect={handleToggleDetect}
                  isCameraOn={camera.isCameraOn}
                />
              </>
            ) : (
              <div className="learned-view" id="learned-view">
                <div className="learned-section">
                  <div className="learned-section-title">
                    👤 Learned Faces
                  </div>
                  {learnedItems?.faces?.length > 0 ? (
                    <div className="learned-grid">
                      {learnedItems.faces.map((face, i) => (
                        <div className="learned-card" key={`face-${i}`}>
                          <div className="learned-card-icon">👤</div>
                          <div className="learned-card-label">
                            {face.name || face.label || face}
                          </div>
                          {face.learned_at && (
                            <div className="learned-card-time">
                              {new Date(face.learned_at).toLocaleDateString()}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="learned-empty">
                      No faces learned yet. Start detecting to teach LUNA!
                    </div>
                  )}
                </div>

                <div className="learned-section">
                  <div className="learned-section-title">
                    📦 Learned Objects
                  </div>
                  {learnedItems?.objects?.length > 0 ? (
                    <div className="learned-grid">
                      {learnedItems.objects.map((obj, i) => (
                        <div className="learned-card" key={`obj-${i}`}>
                          <div className="learned-card-icon">📦</div>
                          <div className="learned-card-label">
                            {obj.name || obj.label || obj}
                          </div>
                          {obj.learned_at && (
                            <div className="learned-card-time">
                              {new Date(obj.learned_at).toLocaleDateString()}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="learned-empty">
                      No objects learned yet. LUNA will ask when she sees something new!
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Learning Modal */}
      {learningItem && (
        <LearningModal
          item={learningItem}
          onSubmit={handleLearnSubmit}
          onCancel={() => setLearningItem(null)}
        />
      )}
    </div>
  );
}

export default App;
