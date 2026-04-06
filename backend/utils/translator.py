"""
Project LUNA — Multi-Language Translator
Translates detection results into English, Tamil, and Hindi.
Uses a local dictionary for reliability with optional API fallback.
"""

import logging

logger = logging.getLogger(__name__)


# ---- Local Translation Dictionaries ----
# Common detection terms pre-translated for reliability

TRANSLATIONS = {
    "ta": {  # Tamil
        # Emotions
        "happy": "மகிழ்ச்சி",
        "sad": "சோகம்",
        "angry": "கோபம்",
        "neutral": "நடுநிலை",
        "surprise": "ஆச்சரியம்",
        "fear": "பயம்",
        "disgust": "வெறுப்பு",

        # Common objects
        "person": "நபர்",
        "bottle": "பாட்டில்",
        "laptop": "மடிக்கணினி",
        "phone": "தொலைபேசி",
        "book": "புத்தகம்",
        "cup": "கோப்பை",
        "chair": "நாற்காலி",
        "keyboard": "விசைப்பலகை",
        "mouse": "சுட்டி",
        "monitor": "திரை",
        "car": "கார்",
        "dog": "நாய்",
        "cat": "பூனை",
        "table": "மேசை",
        "pen": "பேனா",
        "bag": "பை",
        "watch": "கடிகாரம்",
        "glass": "கண்ணாடி",

        # Signs
        "hello": "வணக்கம்",
        "yes": "ஆம்",
        "no": "இல்லை",
        "thank_you": "நன்றி",
        "help": "உதவி",
        "i_love_you": "நான் உன்னை நேசிக்கிறேன்",

        # UI phrases
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
    "hi": {  # Hindi
        # Emotions
        "happy": "खुश",
        "sad": "दुखी",
        "angry": "गुस्सा",
        "neutral": "तटस्थ",
        "surprise": "आश्चर्य",
        "fear": "डर",
        "disgust": "घृणा",

        # Common objects
        "person": "व्यक्ति",
        "bottle": "बोतल",
        "laptop": "लैपटॉप",
        "phone": "फ़ोन",
        "book": "किताब",
        "cup": "कप",
        "chair": "कुर्सी",
        "keyboard": "कीबोर्ड",
        "mouse": "माउस",
        "monitor": "मॉनिटर",
        "car": "कार",
        "dog": "कुत्ता",
        "cat": "बिल्ली",
        "table": "मेज",
        "pen": "कलम",
        "bag": "बैग",
        "watch": "घड़ी",
        "glass": "गिलास",

        # Signs
        "hello": "नमस्ते",
        "yes": "हाँ",
        "no": "नहीं",
        "thank_you": "धन्यवाद",
        "help": "मदद",
        "i_love_you": "मैं तुमसे प्यार करता हूँ",

        # UI phrases
        "unknown": "अज्ञात",
        "detected": "पता चला",
        "face": "चेहरा",
        "object": "वस्तु",
        "emotion": "भावना",
        "sign": "संकेत",
        "who_is_this": "यह कौन है?",
        "i_see": "मैं देख रहा हूँ",
        "looking": "देख रहा है",
    }
}


class Translator:
    """Multi-language translator with local dictionary."""

    def __init__(self):
        self.dictionaries = TRANSLATIONS
        self.googletrans_available = False
        self._try_load_googletrans()

    def _try_load_googletrans(self):
        """Try to load googletrans for dynamic translation fallback."""
        try:
            from googletrans import Translator as GTranslator
            self.gtrans = GTranslator()
            self.googletrans_available = True
            logger.info("✅ googletrans available for dynamic translations")
        except ImportError:
            self.gtrans = None
            logger.info("ℹ️ Using local dictionary translations only")

    def translate_word(self, word, lang):
        """
        Translate a single word/phrase.

        Args:
            word: English word to translate
            lang: Target language code ('ta' for Tamil, 'hi' for Hindi)

        Returns:
            str: Translated word, or original if not found
        """
        if lang == "en":
            return word

        word_lower = word.lower().strip()
        dict_for_lang = self.dictionaries.get(lang, {})

        if word_lower in dict_for_lang:
            return dict_for_lang[word_lower]

        # Try googletrans fallback
        if self.googletrans_available:
            try:
                result = self.gtrans.translate(word, dest=lang)
                return result.text
            except Exception:
                pass

        return word  # Return original if no translation

    def translate_message(self, message, lang):
        """
        Translate a full message. Uses word-by-word dictionary lookup
        with googletrans fallback for the full sentence.

        Args:
            message: English message to translate
            lang: Target language code

        Returns:
            str: Translated message
        """
        if lang == "en":
            return message

        # Try full sentence translation with googletrans first
        if self.googletrans_available:
            try:
                result = self.gtrans.translate(message, dest=lang)
                return result.text
            except Exception:
                pass

        # Fallback: word-by-word replacement
        translated = message
        dict_for_lang = self.dictionaries.get(lang, {})

        for eng, trans in dict_for_lang.items():
            # Case-insensitive replacement
            import re
            translated = re.sub(
                re.escape(eng), trans, translated, flags=re.IGNORECASE
            )

        return translated

    def translate_detection_results(self, results, lang):
        """
        Translate detection result labels.

        Args:
            results: Detection results dict
            lang: Target language code

        Returns:
            dict: Results with translated labels
        """
        if lang == "en":
            return results

        translated = {}

        # Translate object labels
        if "objects" in results:
            translated["objects"] = [
                {**obj, "label_translated": self.translate_word(obj["label"], lang)}
                for obj in results["objects"]
            ]

        # Translate face names (names don't translate, but "Unknown" does)
        if "faces" in results:
            translated["faces"] = [
                {
                    **face,
                    "name_translated": self.translate_word("unknown", lang)
                    if face.get("is_unknown") else face["name"]
                }
                for face in results["faces"]
            ]

        # Translate emotions
        if "emotions" in results:
            translated["emotions"] = [
                {**emo, "emotion_translated": self.translate_word(emo["emotion"], lang)}
                for emo in results["emotions"]
            ]

        # Translate signs
        if "signs" in results:
            translated["signs"] = [
                {**sign, "sign_translated": self.translate_word(sign["sign"], lang)}
                for sign in results["signs"]
            ]

        # Translate message
        if "message" in results:
            translated["message"] = self.translate_message(results["message"], lang)

        return translated

    def get_supported_languages(self):
        """Return supported languages."""
        from config import LANGUAGES
        return LANGUAGES
