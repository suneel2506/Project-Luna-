/**
 * Project LUNA — CameraPreview Component
 * Live webcam feed with detection overlay badges.
 */

import { useEffect, useRef } from 'react';

export default function CameraPreview({
  videoRef,
  canvasRef,
  isCameraOn,
  cameraError,
  onStartCamera,
  onStopCamera,
  onCapture,
  isDetecting,
  isProcessing,
  detectionResults,
}) {
  const intervalRef = useRef(null);

  // Auto-capture frames when detecting
  useEffect(() => {
    if (isDetecting && isCameraOn) {
      // Capture immediately
      onCapture();
      // Then every 2 seconds
      intervalRef.current = setInterval(() => {
        onCapture();
      }, 2000);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isDetecting, isCameraOn, onCapture]);

  const badges = [];
  if (detectionResults) {
    if (detectionResults.objects?.length) {
      detectionResults.objects.forEach((obj, i) => {
        badges.push(
          <span key={`obj-${i}`} className="detection-badge badge-object">
            📦 {obj.label} {obj.confidence ? `${Math.round(obj.confidence * 100)}%` : ''}
          </span>
        );
      });
    }
    if (detectionResults.faces?.length) {
      detectionResults.faces.forEach((face, i) => {
        badges.push(
          <span key={`face-${i}`} className="detection-badge badge-face">
            👤 {face.name || 'Unknown'}
          </span>
        );
      });
    }
    if (detectionResults.emotions?.length) {
      detectionResults.emotions.forEach((emo, i) => {
        badges.push(
          <span key={`emo-${i}`} className="detection-badge badge-emotion">
            😊 {emo.emotion || emo}
          </span>
        );
      });
    }
    if (detectionResults.signs?.length) {
      detectionResults.signs.forEach((sign, i) => {
        badges.push(
          <span key={`sign-${i}`} className="detection-badge badge-sign">
            🤟 {sign.sign || sign}
          </span>
        );
      });
    }
  }

  return (
    <div className="camera-panel" id="camera-panel">
      <div className="camera-header">
        <div className="camera-title">
          <span>📷</span>
          <span>Camera Feed</span>
          {isCameraOn && (
            <span className="status-dot" style={{ marginLeft: 4 }} />
          )}
        </div>
        <div className="camera-controls">
          {!isCameraOn ? (
            <button
              className="cam-btn cam-btn-start"
              onClick={onStartCamera}
              id="btn-start-camera"
            >
              ▶ Start
            </button>
          ) : (
            <>
              <button
                className="cam-btn cam-btn-stop"
                onClick={onStopCamera}
                id="btn-stop-camera"
              >
                ⏹ Stop
              </button>
            </>
          )}
        </div>
      </div>

      <div className="camera-viewport">
        {isCameraOn ? (
          <>
            <video
              ref={videoRef}
              className="camera-video"
              autoPlay
              playsInline
              muted
              id="camera-video"
            />
            <canvas ref={canvasRef} className="camera-canvas" />
            {badges.length > 0 && (
              <div className="detection-badges">{badges}</div>
            )}
            {isProcessing && (
              <div
                className="detection-badge badge-object"
                style={{
                  position: 'absolute',
                  bottom: 12,
                  right: 12,
                  animation: 'pulse 1s infinite',
                }}
              >
                ⏳ Analyzing...
              </div>
            )}
          </>
        ) : (
          <div className="camera-placeholder">
            <div className="camera-placeholder-icon">📷</div>
            <div className="camera-placeholder-text">
              Click <strong>Start</strong> to activate your camera and begin AI vision detection
            </div>
          </div>
        )}
        {cameraError && <div className="camera-error">{cameraError}</div>}
      </div>
    </div>
  );
}
