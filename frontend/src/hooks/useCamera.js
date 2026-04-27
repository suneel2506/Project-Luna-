/**
 * Project LUNA — useCamera Hook
 * Custom hook for webcam access and frame capture.
 *
 * The video element is ALWAYS in the DOM (hidden when off), so
 * videoRef.current is always available for stream attachment.
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
    if (!video || !stream) {
      console.warn('[useCamera] attachStream: video or stream is null', {
        hasVideo: !!video,
        hasStream: !!stream,
      });
      return false;
    }

    // Always re-assign to handle React re-mounts
    video.srcObject = stream;

    // If metadata is already available, play immediately
    if (video.readyState >= 1) {
      try {
        await video.play();
      } catch {
        /* ignore autoplay errors */
      }
      return true;
    }

    // Wait for metadata to load
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        // Don't reject — the stream might still work, just slowly
        console.warn('[useCamera] Stream metadata took > 10s');
        resolve();
      }, 10000);

      video.onloadedmetadata = () => {
        clearTimeout(timeout);
        video.play().then(resolve).catch((e) => {
          console.warn('[useCamera] video.play() failed:', e);
          resolve(); // Don't reject — autoplay policies may block
        });
      };

      video.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('Video element error'));
      };
    });

    return true;
  }, []);

  // ── Re-sync effect: ensure video element always has the stream ──
  useEffect(() => {
    const stream = streamRef.current;
    const video = videoRef.current;

    if (stream && stream.active && video && isCameraOn) {
      if (video.srcObject !== stream) {
        console.log('[useCamera] Re-syncing stream to video element');
        video.srcObject = stream;
        video.play().catch(() => {});
      } else if (video.paused) {
        video.play().catch(() => {});
      }
    }
  }); // runs every render — cheap ref check

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
        const attached = await attachStream(streamRef.current);
        setIsCameraOn(true);
        setCameraReady(true);
        return attached;
      }

      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;

      // Attach stream to video element (element is always in DOM now)
      const attached = await attachStream(stream);
      if (!attached) {
        console.warn('[useCamera] attachStream returned false, will re-sync on render');
      }

      setIsCameraOn(true);
      setCameraReady(true);
      return true;
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
      return false;
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
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !isCameraOn) {
      return null;
    }

    // Don't capture if video isn't actually playing with valid frames
    if (video.readyState < 2 || video.videoWidth === 0) {
      return null;
    }

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Quick check: is the frame all black? (camera warming up)
    const sample = ctx.getImageData(0, 0, 20, 20);
    const px = sample.data;
    let hasColor = false;
    for (let i = 0; i < px.length; i += 4) {
      if (px[i] > 10 || px[i + 1] > 10 || px[i + 2] > 10) {
        hasColor = true;
        break;
      }
    }

    if (!hasColor) {
      return null; // Skip black frames silently
    }

    return canvas.toDataURL('image/jpeg', 0.7);
  }, [isCameraOn]);

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
