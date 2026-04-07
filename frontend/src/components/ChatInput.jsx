/**
 * Project LUNA — ChatInput Component
 * Text input with send button and detection toggle.
 */

import { useState } from 'react';

export default function ChatInput({
  onSendMessage,
  isDetecting,
  onToggleDetect,
  isCameraOn,
  disabled,
}) {
  const [text, setText] = useState('');

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSendMessage(trimmed);
    setText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-input-area" id="chat-input-area">
      <div className="chat-input-row">
        {/* Detection toggle */}
        <button
          className={`cam-btn ${isDetecting ? 'cam-btn-stop' : 'cam-btn-detect'}`}
          onClick={onToggleDetect}
          disabled={!isCameraOn}
          title={isDetecting ? 'Stop Detection' : 'Start Detection'}
          id="btn-toggle-detect"
          style={{ height: 44, whiteSpace: 'nowrap' }}
        >
          {isDetecting ? '⏹ Stop' : '🔍 Detect'}
        </button>

        {/* Text input */}
        <div className="chat-input-wrapper">
          <input
            className="chat-input"
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message to LUNA..."
            disabled={disabled}
            id="chat-text-input"
          />
        </div>

        {/* Send button */}
        <button
          className="chat-send-btn"
          onClick={handleSend}
          disabled={!text.trim() || disabled}
          aria-label="Send"
          id="btn-send-message"
        >
          ➤
        </button>
      </div>
    </div>
  );
}
