/**
 * Project LUNA — useCamera Hook
 * Custom hook for webcam access and frame capture.
 *
 * Handles React StrictMode double-mount gracefully by re-syncing
 * the video element with any existing stream after re-mount.
 */

import { useState, useRef, useCallback, useEffect } from 'react';

export function useCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);

  // ── Helper: attach a stream to the current video element ──
  const attachStream = useCallback(async (stream) => {
    const video = videoRef.current;
    if (!video || !stream) return false;

    // Already attached to this exact element — just make sure it's playing
    if (video.srcObject === stream) {
      if (video.paused) {
        try { await video.play(); } catch { /* ignore */ }
      }
      return true;
    }

    video.srcObject = stream;

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Camera stream timeout — try refreshing the page.'));
      }, 10000);

      // If metadata is already available (re-mount scenario), resolve immediately
      if (video.readyState >= 1) {
        clearTimeout(timeout);
        video.play().then(resolve).catch(reject);
        return;
      }

      video.onloadedmetadata = () => {
        clearTimeout(timeout);
        video.play().then(resolve).catch(reject);
      };

      video.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('Video element error'));
      };
    });

    return true;
  }, []);

  // ── Re-sync effect: if we already have a live stream but the video
  //    element changed (React StrictMode unmount/re-mount), re-attach it ──
  useEffect(() => {
    const stream = streamRef.current;
    const video = videoRef.current;

    if (stream && video && isCameraOn) {
      if (video.srcObject !== stream) {
        attachStream(stream)
          .then(() => setCameraReady(true))
          .catch((err) => {
            console.warn('Camera re-sync failed:', err);
            setCameraReady(false);
          });
      }
    }
  }); // runs every render — cheap ref check, only acts when needed

  const startCamera = useCallback(async () => {
    try {
      setCameraError(null);
      setCameraReady(false);

      // Check if getUserMedia is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API not available. Use HTTPS or localhost.');
      }

      // If we already have a live stream, just re-attach
      if (streamRef.current && streamRef.current.active) {
        await attachStream(streamRef.current);
        setIsCameraOn(true);
        setCameraReady(true);
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;

      await attachStream(stream);
      setCameraReady(true);
      setIsCameraOn(true);
    } catch (err) {
      console.error('Camera access failed:', err);
      let errorMessage;

      if (err.name === 'NotAllowedError') {
        errorMessage = '🔒 Camera access denied. Click the camera icon in your browser\'s address bar to allow access, then try again.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMessage = '📷 No camera found. Please connect a webcam and try again.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMessage = '⚠️ Camera is in use by another app. Close other apps using the camera and try again.';
      } else if (err.name === 'OverconstrainedError') {
        errorMessage = '⚠️ Camera doesn\'t support requested resolution. Trying again...';
      } else {
        errorMessage = `⚠️ Camera error: ${err.message}`;
      }

      setCameraError(errorMessage);
      setIsCameraOn(false);
      setCameraReady(false);
    }
  }, [attachStream]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
    setCameraReady(false);
  }, []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !isCameraOn || !cameraReady) {
      return null;
    }

    const video = videoRef.current;

    // Don't capture if video isn't actually playing
    if (video.readyState < 2 || video.videoWidth === 0) {
      return null;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Check if frame is actually valid (not all black)
    const imageData = ctx.getImageData(0, 0, 10, 10);
    const pixels = imageData.data;
    let isBlack = true;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10) {
        isBlack = false;
        break;
      }
    }

    if (isBlack) {
      return null; // Skip black frames
    }

    return canvas.toDataURL('image/jpeg', 0.7);
  }, [isCameraOn, cameraReady]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return {
    videoRef,
    canvasRef,
    isCameraOn,
    cameraError,
    cameraReady,
    startCamera,
    stopCamera,
    captureFrame,
  };
}
