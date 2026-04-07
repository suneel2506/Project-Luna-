/**
 * Project LUNA — LearningModal Component
 * Modal dialog for labeling unknown faces/objects.
 */

import { useState } from 'react';

export default function LearningModal({ item, onSubmit, onCancel }) {
  const [label, setLabel] = useState('');

  if (!item) return null;

  const isFace = item.type === 'face';

  const handleSubmit = () => {
    const trimmed = label.trim();
    if (!trimmed) return;
    onSubmit(item.type, item.id, trimmed);
    setLabel('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
    if (e.key === 'Escape') {
      onCancel();
    }
  };

  return (
    <div className="modal-backdrop" onClick={onCancel} id="learning-modal-backdrop">
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        id="learning-modal"
      >
        <div className="modal-title">
          <span>{isFace ? '👤' : '📦'}</span>
          <span>
            {isFace ? 'New Face Detected!' : 'Unknown Object Detected!'}
          </span>
        </div>
        <div className="modal-description">
          {isFace
            ? "I see someone I don't recognize. What's their name?"
            : "I found something I haven't seen before. What is it?"
          }
        </div>

        {item.crop_base64 && (
          <img
            className="modal-image"
            src={
              item.crop_base64.startsWith('data:')
                ? item.crop_base64
                : `data:image/jpeg;base64,${item.crop_base64}`
            }
            alt="Unknown detection"
          />
        )}

        <input
          className="modal-input"
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isFace ? 'Enter their name...' : 'Enter object name...'}
          autoFocus
          id="learning-input"
        />

        <div className="modal-actions">
          <button
            className="modal-btn modal-btn-secondary"
            onClick={onCancel}
            id="btn-cancel-learn"
          >
            Skip
          </button>
          <button
            className="modal-btn modal-btn-primary"
            onClick={handleSubmit}
            disabled={!label.trim()}
            id="btn-submit-learn"
          >
            {isFace ? 'Remember Face' : 'Learn Object'}
          </button>
        </div>
      </div>
    </div>
  );
}
