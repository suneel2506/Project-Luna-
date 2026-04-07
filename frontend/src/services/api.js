/**
 * Project LUNA — API Service
 * Axios client for communicating with the Flask backend.
 */

import axios from 'axios';
import { API_BASE_URL } from '../utils/constants';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Process a camera frame through all AI detectors.
 * @param {string} imageData - Base64 encoded image
 * @param {string} language - Language code (en, ta, hi)
 * @param {object} detectors - Which detectors to enable
 * @returns {Promise<object>} Detection results
 */
export async function processFrame(imageData, language = 'en', detectors = {}) {
  const response = await api.post('/api/process', {
    image: imageData,
    language,
    detectors: {
      objects: true,
      faces: true,
      emotions: true,
      signs: true,
      ...detectors,
    },
  });
  return response.data;
}

/**
 * Submit a user label for an unknown detection.
 * @param {string} type - 'face' or 'object'
 * @param {string} id - The unknown item ID
 * @param {string} label - User-provided label
 * @returns {Promise<object>} Learn result
 */
export async function submitLabel(type, id, label) {
  const response = await api.post('/api/learn', { type, id, label });
  return response.data;
}

/**
 * Get health/status of the LUNA backend.
 * @returns {Promise<object>} Status info
 */
export async function getStatus() {
  const response = await api.get('/api/status');
  return response.data;
}

/**
 * Get learned items history.
 * @returns {Promise<object>} Learned summary
 */
export async function getHistory() {
  const response = await api.get('/api/history');
  return response.data;
}

/**
 * Get supported languages.
 * @returns {Promise<object>} Language list
 */
export async function getLanguages() {
  const response = await api.get('/api/languages');
  return response.data;
}

export default api;
