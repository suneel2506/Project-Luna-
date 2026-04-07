/**
 * Project LUNA — ChatWindow Component
 * Scrollable message list with auto-scroll and typing indicator.
 */

import ChatMessage from './ChatMessage';

export default function ChatWindow({ messages, scrollRef, isProcessing }) {
  return (
    <div className="chat-messages" ref={scrollRef} id="chat-messages">
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} />
      ))}

      {isProcessing && (
        <div className="message ai">
          <div className="message-avatar ai-avatar">🌙</div>
          <div className="message-bubble">
            <div className="typing-indicator">
              <div className="typing-dot" />
              <div className="typing-dot" />
              <div className="typing-dot" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
