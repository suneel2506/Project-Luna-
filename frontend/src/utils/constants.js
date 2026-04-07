/**
 * Project LUNA — App Constants
 */

export const API_BASE_URL = 'http://localhost:5000';

export const FRAME_INTERVAL_MS = 2000;

export const LANGUAGES = [
  { code: 'en', label: 'English', native: 'EN' },
  { code: 'ta', label: 'Tamil', native: 'தமிழ்' },
  { code: 'hi', label: 'Hindi', native: 'हिंदी' },
];

export const DETECTOR_OPTIONS = [
  { key: 'objects', label: 'Objects', icon: '📦', description: 'Detect objects using YOLO' },
  { key: 'faces', label: 'Faces', icon: '👤', description: 'Recognize faces' },
  { key: 'emotions', label: 'Emotions', icon: '😊', description: 'Detect facial emotions' },
  { key: 'signs', label: 'Signs', icon: '🤟', description: 'Sign language detection' },
];

export const MESSAGE_TYPES = {
  USER: 'user',
  AI: 'ai',
  SYSTEM: 'system',
  LEARNING: 'learning',
};

export const NAV_ITEMS = [
  { key: 'chat', label: 'Chat', icon: '💬' },
  { key: 'learned', label: 'Learned', icon: '🧠' },
  { key: 'settings', label: 'Settings', icon: '⚙️' },
];
