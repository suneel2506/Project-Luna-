// PASTE THIS INTO YOUR BROWSER CONSOLE (F12 -> Console tab)
// to diagnose the camera issue

(async function debugCamera() {
  console.log('=== LUNA Camera Debug ===');
  
  // 1. Check video element
  const video = document.getElementById('camera-video');
  if (!video) {
    console.log('❌ No video element found on page. Camera may not be started.');
    return;
  }
  
  console.log('Video element:', {
    readyState: video.readyState,
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    paused: video.paused,
    hasSrcObject: !!video.srcObject,
    currentTime: video.currentTime,
    offsetWidth: video.offsetWidth,
    offsetHeight: video.offsetHeight,
    display: getComputedStyle(video).display,
    visibility: getComputedStyle(video).visibility,
    opacity: getComputedStyle(video).opacity,
  });
  
  // 2. Check if stream has active tracks
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    console.log('Stream tracks:', tracks.map(t => ({
      kind: t.kind,
      label: t.label,
      enabled: t.enabled,
      readyState: t.readyState,
      muted: t.muted,
    })));
  } else {
    console.log('❌ No srcObject on video element!');
  }
  
  // 3. Try direct camera access
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const tracks = stream.getTracks();
    console.log('✅ Direct camera access works!', tracks.map(t => t.label));
    
    // Test: force the stream onto the video element
    video.srcObject = stream;
    await video.play();
    console.log('✅ Video is playing with new stream!');
    console.log('New dimensions:', video.videoWidth, 'x', video.videoHeight);
  } catch(e) {
    console.log('❌ Camera access failed:', e.message);
  }
  
  console.log('=== Debug Complete ===');
})();
