/**
 * Project LUNA — Header Component
 * App header with LUNA branding, status indicator, and language selector.
 */

import { LANGUAGES } from '../utils/constants';

export default function Header({
  language,
  onLanguageChange,
  backendStatus,
  isProcessing,
  onToggleSidebar,
}) {
  const statusClass = isProcessing
    ? 'processing'
    : backendStatus === 'online'
    ? ''
    : 'offline';

  const statusText = isProcessing
    ? 'Processing...'
    : backendStatus === 'online'
    ? 'Connected'
    : 'Offline';

  return (
    <header className="header" id="app-header">
      <div className="header-left">
        <button
          className="header-burger"
          onClick={onToggleSidebar}
          aria-label="Toggle sidebar"
          id="sidebar-toggle"
        >
          ☰
        </button>
        <div className="header-status">
          <span className={`status-dot ${statusClass}`} />
          <span className="status-text">{statusText}</span>
        </div>
      </div>

      <div className="header-right">
        <div className="language-selector" id="language-selector">
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              className={`lang-btn ${language === lang.code ? 'active' : ''}`}
              onClick={() => onLanguageChange(lang.code)}
              title={lang.label}
              id={`lang-${lang.code}`}
            >
              {lang.native}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
