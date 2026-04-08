/**
 * Project LUNA — ChatMessage Component
 * Individual chat message bubble with detection results and translations.
 */

import { MESSAGE_TYPES } from '../utils/constants';

function formatTime(timestamp) {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

export default function ChatMessage({ message }) {
  const { type, content, timestamp, detectionData } = message;

  const typeClass =
    type === MESSAGE_TYPES.USER
      ? 'user'
      : type === MESSAGE_TYPES.AI
      ? 'ai'
      : type === MESSAGE_TYPES.LEARNING
      ? 'learning'
      : 'system';

  const avatar =
    type === MESSAGE_TYPES.AI ? (
      <div className="message-avatar ai-avatar">🌙</div>
    ) : type === MESSAGE_TYPES.USER ? (
      <div className="message-avatar user-avatar">👤</div>
    ) : type === MESSAGE_TYPES.LEARNING ? (
      <div className="message-avatar ai-avatar">🧠</div>
    ) : null;

  // Build detection chips
  const chips = [];
  if (detectionData) {
    if (detectionData.objects?.length) {
      detectionData.objects.forEach((obj, i) => {
        chips.push(
          <span key={`obj-${i}`} className="detection-chip">
            📦 {obj.label}
          </span>
        );
      });
    }
    if (detectionData.faces?.length) {
      detectionData.faces.forEach((face, i) => {
        chips.push(
          <span key={`face-${i}`} className="detection-chip">
            👤 {face.name}
          </span>
        );
      });
    }
    if (detectionData.emotions?.length) {
      detectionData.emotions.forEach((emo, i) => {
        chips.push(
          <span key={`emo-${i}`} className="detection-chip">
            😊 {emo.emotion || emo}
          </span>
        );
      });
    }
    if (detectionData.signs?.length) {
      detectionData.signs.forEach((sign, i) => {
        chips.push(
          <span key={`sign-${i}`} className="detection-chip">
            🤟 {sign.sign || sign}
          </span>
        );
      });
    }
  }

  // Translations — only show non-English translations that differ from the main message
  const translations = detectionData?.translations;
  const hasTranslations =
    translations &&
    type === MESSAGE_TYPES.AI &&
    (translations.ta || translations.hi);

  // Check if translations are actually different from English
  const showTamil =
    translations?.ta && translations.ta !== translations?.en && translations.ta !== content;
  const showHindi =
    translations?.hi && translations.hi !== translations?.en && translations.hi !== content;

  return (
    <div className={`message ${typeClass}`}>
      {avatar}
      <div>
        <div className="message-bubble">
          {content}
          {chips.length > 0 && <div className="detection-results">{chips}</div>}
          {hasTranslations && (showTamil || showHindi) && (
            <div className="translations">
              {showTamil && (
                <div className="translation-row">
                  <span className="translation-label">தமிழ்: </span>
                  {translations.ta}
                </div>
              )}
              {showHindi && (
                <div className="translation-row">
                  <span className="translation-label">हिंदी: </span>
                  {translations.hi}
                </div>
              )}
            </div>
          )}
        </div>
        <div className="message-time">{formatTime(timestamp)}</div>
      </div>
    </div>
  );
}
