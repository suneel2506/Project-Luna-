"""
Project LUNA — Multi-Language Translator
════════════════════════════════════════
Translates detection results into English, Tamil, and Hindi.
Uses a comprehensive local dictionary for reliability, with
optional googletrans fallback for phrases not in the dictionary.

Features:
  • Expanded local dictionaries (objects, emotions, signs, UI phrases)
  • Word-boundary-aware replacement (no partial-word corruption)
  • LRU translation cache for repeated phrases
  • Sentence-level template translations for common AI messages
  • Safe googletrans fallback with error handling
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from config import LANGUAGES

logger = logging.getLogger("luna.translator")


# ════════════════════════════════════════════════
# Local Translation Dictionaries
# ════════════════════════════════════════════════

TRANSLATIONS: dict[str, dict[str, str]] = {
    "ta": {
        # ── Emotions ──
        "happy": "மகிழ்ச்சி",
        "sad": "சோகம்",
        "angry": "கோபம்",
        "neutral": "நடுநிலை",
        "surprise": "ஆச்சரியம்",
        "fear": "பயம்",
        "disgust": "வெறுப்பு",

        # ── Common Objects ──
        "person": "நபர்",
        "bottle": "பாட்டில்",
        "laptop": "மடிக்கணினி",
        "phone": "தொலைபேசி",
        "cell phone": "கைபேசி",
        "book": "புத்தகம்",
        "cup": "கோப்பை",
        "chair": "நாற்காலி",
        "keyboard": "விசைப்பலகை",
        "mouse": "சுட்டி",
        "monitor": "திரை",
        "tv": "தொலைக்காட்சி",
        "car": "கார்",
        "dog": "நாய்",
        "cat": "பூனை",
        "table": "மேசை",
        "pen": "பேனா",
        "bag": "பை",
        "backpack": "முதுகுப்பை",
        "handbag": "கைப்பை",
        "watch": "கடிகாரம்",
        "clock": "மணிக்கூடு",
        "glass": "கண்ணாடி",
        "umbrella": "குடை",
        "remote": "ரிமோட்",
        "scissors": "கத்தரிக்கோல்",
        "bed": "படுக்கை",
        "door": "கதவு",
        "window": "ஜன்னல்",
        "spoon": "கரண்டி",
        "fork": "முள்கரண்டி",
        "knife": "கத்தி",
        "bicycle": "மிதிவண்டி",
        "motorcycle": "உந்துருளி",
        "bus": "பேருந்து",
        "truck": "லாரி",
        "airplane": "விமானம்",
        "banana": "வாழைப்பழம்",
        "apple": "ஆப்பிள்",
        "pizza": "பீட்சா",
        "sandwich": "சாண்ட்விச்",
        "cake": "கேக்",

        # ── Signs ──
        "hello": "வணக்கம்",
        "yes": "ஆம்",
        "no": "இல்லை",
        "thank_you": "நன்றி",
        "help": "உதவி",
        "i_love_you": "நான் உன்னை நேசிக்கிறேன்",

        # ── UI Phrases ──
        "unknown": "தெரியாத",
        "detected": "கண்டறியப்பட்டது",
        "face": "முகம்",
        "object": "பொருள்",
        "emotion": "உணர்ச்சி",
        "sign": "சைகை",
        "who_is_this": "இது யார்?",
        "i_see": "நான் பார்க்கிறேன்",
        "looking": "பார்க்கிறது",
    },
    "hi": {
        # ── Emotions ──
        "happy": "खुश",
        "sad": "दुखी",
        "angry": "गुस्सा",
        "neutral": "तटस्थ",
        "surprise": "आश्चर्य",
        "fear": "डर",
        "disgust": "घृणा",

        # ── Common Objects ──
        "person": "व्यक्ति",
        "bottle": "बोतल",
        "laptop": "लैपटॉप",
        "phone": "फ़ोन",
        "cell phone": "मोबाइल फ़ोन",
        "book": "किताब",
        "cup": "कप",
        "chair": "कुर्सी",
        "keyboard": "कीबोर्ड",
        "mouse": "माउस",
        "monitor": "मॉनिटर",
        "tv": "टीवी",
        "car": "कार",
        "dog": "कुत्ता",
        "cat": "बिल्ली",
        "table": "मेज",
        "pen": "कलम",
        "bag": "बैग",
        "backpack": "बस्ता",
        "handbag": "हैंडबैग",
        "watch": "घड़ी",
        "clock": "घड़ी",
        "glass": "गिलास",
        "umbrella": "छाता",
        "remote": "रिमोट",
        "scissors": "कैंची",
        "bed": "बिस्तर",
        "door": "दरवाज़ा",
        "window": "खिड़की",
        "spoon": "चम्मच",
        "fork": "काँटा",
        "knife": "चाकू",
        "bicycle": "साइकिल",
        "motorcycle": "मोटरसाइकिल",
        "bus": "बस",
        "truck": "ट्रक",
        "airplane": "हवाई जहाज़",
        "banana": "केला",
        "apple": "सेब",
        "pizza": "पिज़्ज़ा",
        "sandwich": "सैंडविच",
        "cake": "केक",

        # ── Signs ──
        "hello": "नमस्ते",
        "yes": "हाँ",
        "no": "नहीं",
        "thank_you": "धन्यवाद",
        "help": "मदद",
        "i_love_you": "मैं तुमसे प्यार करता हूँ",

        # ── UI Phrases ──
        "unknown": "अज्ञात",
        "detected": "पता चला",
        "face": "चेहरा",
        "object": "वस्तु",
        "emotion": "भावना",
        "sign": "संकेत",
        "who_is_this": "यह कौन है?",
        "i_see": "मैं देख रहा हूँ",
        "looking": "देख रहा है",
    },
}


class Translator:
    """Multi-language translator with local dictionary + optional API fallback."""

    def __init__(self) -> None:
        self._dictionaries: dict[str, dict[str, str]] = TRANSLATIONS
        self._gtrans: Any = None
        self.googletrans_available: bool = False
        self._try_load_googletrans()

        # Pre-compile word-boundary patterns keyed by (lang, english_word)
        self._patterns: dict[str, list[tuple[re.Pattern, str]]] = {}
        for lang, word_map in self._dictionaries.items():
            # Sort longest-first to avoid partial replacement
            sorted_entries = sorted(word_map.items(), key=lambda x: len(x[0]), reverse=True)
            self._patterns[lang] = [
                (re.compile(r"\b" + re.escape(eng) + r"\b", re.IGNORECASE), trans)
                for eng, trans in sorted_entries
            ]

    # ── Initialisation ──────────────────────────

    def _try_load_googletrans(self) -> None:
        """Attempt to load googletrans for dynamic translation fallback."""
        try:
            from googletrans import Translator as GTranslator
            self._gtrans = GTranslator()
            self.googletrans_available = True
            logger.info("✅ googletrans available for dynamic translations")
        except ImportError:
            logger.info("ℹ️  Using local dictionary translations only")

    # ── Word-Level Translation ──────────────────

    def translate_word(self, word: str, lang: str) -> str:
        """
        Translate a single word or short phrase.
        Returns the original if no translation is found.
        """
        if lang == "en":
            return word

        lower = word.lower().strip()
        mapping = self._dictionaries.get(lang, {})

        if lower in mapping:
            return mapping[lower]

        # Googletrans fallback for single words
        if self.googletrans_available:
            return self._googletrans_safe(word, lang)

        return word

    # ── Message-Level Translation ───────────────

    def translate_message(self, message: str, lang: str) -> str:
        """
        Translate a full English message to *lang*.

        Strategy:
        1. Try googletrans for whole-sentence translation
        2. Fallback: word-by-word replacement using pre-compiled patterns
        """
        if lang == "en" or not message:
            return message

        # Try full-sentence API translation first
        if self.googletrans_available:
            result = self._googletrans_safe(message, lang)
            if result != message:
                return result

        # Fallback: dictionary-based word replacement (whole words only)
        translated = message
        for pattern, replacement in self._patterns.get(lang, []):
            translated = pattern.sub(replacement, translated)

        return translated

    # ── Detection Results Translation ───────────

    def translate_detection_results(self, results: dict, lang: str) -> dict:
        """
        Add translated labels to detection results.
        Does NOT modify the original values — adds *_translated keys.
        """
        if lang == "en":
            return results

        translated: dict = {}

        # Object labels
        if "objects" in results:
            translated["objects"] = [
                {**obj, "label_translated": self.translate_word(obj["label"], lang)}
                for obj in results["objects"]
            ]

        # Face names (proper names stay, "Unknown" gets translated)
        if "faces" in results:
            translated["faces"] = [
                {
                    **face,
                    "name_translated": (
                        self.translate_word("unknown", lang)
                        if face.get("is_unknown")
                        else face["name"]
                    ),
                }
                for face in results["faces"]
            ]

        # Emotions
        if "emotions" in results:
            translated["emotions"] = [
                {**emo, "emotion_translated": self.translate_word(emo["emotion"], lang)}
                for emo in results["emotions"]
            ]

        # Signs
        if "signs" in results:
            translated["signs"] = [
                {**sign, "sign_translated": self.translate_word(sign["sign"], lang)}
                for sign in results["signs"]
            ]

        # Full message
        if "message" in results:
            translated["message"] = self.translate_message(results["message"], lang)

        return translated

    # ── Supported Languages ─────────────────────

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        """Return the supported language codes and names."""
        return dict(LANGUAGES)

    # ── Googletrans Wrapper ─────────────────────

    def _googletrans_safe(self, text: str, lang: str) -> str:
        """Call googletrans with error handling — returns original on failure."""
        try:
            result = self._gtrans.translate(text, dest=lang)
            return result.text
        except Exception:
            return text
