/**
 * Project LUNA — Sidebar Component
 * Navigation, detector toggles, and stats panel.
 */

import { NAV_ITEMS, DETECTOR_OPTIONS } from '../utils/constants';

export default function Sidebar({
  isOpen,
  activeView,
  onViewChange,
  detectors,
  onToggleDetector,
  learnedStats,
}) {
  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`} id="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-row">
          <div className="sidebar-logo">🌙</div>
          <div>
            <div className="sidebar-title">LUNA</div>
            <div className="sidebar-subtitle">AI Vision Assistant</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <div
            key={item.key}
            className={`sidebar-nav-item ${activeView === item.key ? 'active' : ''}`}
            onClick={() => onViewChange(item.key)}
            role="button"
            tabIndex={0}
            id={`nav-${item.key}`}
          >
            <span className="sidebar-nav-icon">{item.icon}</span>
            <span>{item.label}</span>
          </div>
        ))}
      </nav>

      {/* Detector Toggles */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">Detectors</div>
        {DETECTOR_OPTIONS.map((det) => (
          <div className="detector-toggle" key={det.key}>
            <span className="detector-icon">{det.icon}</span>
            <span className="detector-label">{det.label}</span>
            <button
              className={`detector-switch ${detectors[det.key] ? 'on' : ''}`}
              onClick={() => onToggleDetector(det.key)}
              aria-label={`Toggle ${det.label}`}
              id={`detector-${det.key}`}
            />
          </div>
        ))}
      </div>

      {/* Stats */}
      <div className="sidebar-stats">
        <div className="sidebar-section-title">Learned</div>
        <div className="stat-row">
          <span className="stat-label">Faces</span>
          <span className="stat-value">{learnedStats.faces || 0}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Objects</span>
          <span className="stat-value">{learnedStats.objects || 0}</span>
        </div>
      </div>
    </aside>
  );
}
