/**
 * Project LUNA — useChat Hook
 * Custom hook for managing chat messages and detection state.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { MESSAGE_TYPES } from '../utils/constants';

let messageIdCounter = 0;

function createMessage(type, content, extra = {}) {
  return {
    id: `msg_${++messageIdCounter}_${Date.now()}`,
    type,
    content,
    timestamp: new Date().toISOString(),
    ...extra,
  };
}

export function useChat() {
  const [messages, setMessages] = useState([
    createMessage(MESSAGE_TYPES.SYSTEM, '🌙 Welcome to LUNA — your AI Vision Assistant! Start your camera and begin detecting.'),
    createMessage(MESSAGE_TYPES.AI, "Hello! I'm **LUNA** 🌙 — your AI Vision Assistant. I can detect objects, recognize faces, read emotions, and understand sign language. Start your camera and click 🔍 **Detect** to see me in action! Type **help** anytime to learn more."),
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const scrollRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const addMessage = useCallback((type, content, extra = {}) => {
    const msg = createMessage(type, content, extra);
    setMessages(prev => [...prev, msg]);
    return msg;
  }, []);

  const addUserMessage = useCallback((content) => {
    return addMessage(MESSAGE_TYPES.USER, content);
  }, [addMessage]);

  const addAIMessage = useCallback((content, detectionData = null) => {
    return addMessage(MESSAGE_TYPES.AI, content, { detectionData });
  }, [addMessage]);

  const addSystemMessage = useCallback((content) => {
    return addMessage(MESSAGE_TYPES.SYSTEM, content);
  }, [addMessage]);

  const addLearningMessage = useCallback((content, learningData = null) => {
    return addMessage(MESSAGE_TYPES.LEARNING, content, { learningData });
  }, [addMessage]);

  const clearMessages = useCallback(() => {
    setMessages([
      createMessage(MESSAGE_TYPES.SYSTEM, '🌙 Chat cleared. Ready for new detections!'),
    ]);
  }, []);

  return {
    messages,
    isProcessing,
    setIsProcessing,
    scrollRef,
    addUserMessage,
    addAIMessage,
    addSystemMessage,
    addLearningMessage,
    clearMessages,
  };
}
