/**
 * Project LUNA — CameraPreview Component
 * Live webcam feed with detection overlay badges.
 *
 * IMPORTANT: The <video> and <canvas> elements are ALWAYS in the DOM
 * so that useCamera can attach the stream before isCameraOn flips to true.
 * They are hidden with CSS when the camera is off.
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
  const onCaptureRef = useRef(onCapture);

  // Keep the ref always pointing to the latest onCapture callback
  useEffect(() => {
    onCaptureRef.current = onCapture;
  }, [onCapture]);

  // NOTE: Camera stream is managed by the useCamera hook.
  // Do NOT call getUserMedia here — it would overwrite the hook's stream
  // and break frame capture.

  // 🔁 AUTO CAPTURE FRAMES (stable interval — not reset by callback changes)
  useEffect(() => {
    if (isDetecting && isCameraOn) {
      // Small delay on first capture to let camera warm up
      const initTimeout = setTimeout(() => {
        onCaptureRef.current();
      }, 500);

      intervalRef.current = setInterval(() => {
        onCaptureRef.current();
      }, 2000);

      return () => {
        clearTimeout(initTimeout);
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isDetecting, isCameraOn]);

  // 🎯 BADGES DISPLAY
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
    <div className="camera-panel">
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
            >
              ▶ Start
            </button>
          ) : (
            <button
              className="cam-btn cam-btn-stop"
              onClick={onStopCamera}
            >
              ⏹ Stop
            </button>
          )}
        </div>
      </div>

      <div className="camera-viewport">
        {/*
          Video & Canvas are ALWAYS in the DOM so the ref is available
          when startCamera() calls attachStream(). Hidden when camera is off.
        */}
        <video
          ref={videoRef}
          id="camera-video"
          className="camera-video"
          autoPlay
          playsInline
          muted
          style={{ display: isCameraOn ? 'block' : 'none' }}
        />
        <canvas ref={canvasRef} className="camera-canvas" />

        {/* Detection overlay badges */}
        {isCameraOn && badges.length > 0 && (
          <div className="detection-badges">{badges}</div>
        )}

        {/* Processing indicator */}
        {isCameraOn && isProcessing && (
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

        {/* Placeholder when camera is off */}
        {!isCameraOn && (
          <div className="camera-placeholder">
            <div className="camera-placeholder-icon">📷</div>
            <div className="camera-placeholder-text">
              Click <strong>Start</strong> to activate your camera
            </div>
          </div>
        )}

        {cameraError && (
          <div className="camera-error">{cameraError}</div>
        )}
      </div>
    </div>
  );
}