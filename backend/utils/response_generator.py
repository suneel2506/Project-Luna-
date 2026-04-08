"""
Project LUNA — Natural Language Response Generator
══════════════════════════════════════════════════
Creates human-like, conversational messages from detection results.
Combines multiple detections into coherent sentences.

Features:
  • Context awareness — avoids repeating the same message back-to-back
  • Deduplication window — suppresses identical messages within N seconds
  • Time-aware greetings (morning / afternoon / evening)
  • Scene-change detection ("The scene has changed!")
  • Smart multi-detection summarisation
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime

from config import DEDUP_WINDOW_SECONDS

logger = logging.getLogger("luna.response_generator")


class ResponseGenerator:
    """Generate natural language responses from AI detection results."""

    def __init__(self) -> None:
        self._last_message: str = ""
        self._last_message_time: float = 0.0
        self._context_objects: set[str] = set()       # labels seen recently
        self._context_faces: set[str] = set()         # names seen recently

    # ════════════════════════════════════════════
    # Response Templates
    # ════════════════════════════════════════════

    GREETING_TEMPLATES: list[str] = [
        "👋 Hello! Let me analyze what I see...",
        "🔍 Scanning the scene now...",
        "👀 Let me take a look...",
        "🌙 LUNA is observing...",
    ]

    OBJECT_TEMPLATES: list[str] = [
        "I can see {article} **{label}** in the frame.",
        "There's {article} **{label}** here.",
        "I detected {article} **{label}** ({confidence}% confident).",
        "I spot {article} **{label}**!",
    ]

    MULTI_OBJECT_TEMPLATES: list[str] = [
        "I can see several things: {items}.",
        "In the frame, I notice: {items}.",
        "Here's what I detect: {items}.",
    ]

    FACE_KNOWN_TEMPLATES: list[str] = [
        "I recognize **{name}**! 👋",
        "Hey, it's **{name}**!",
        "I see **{name}** in the frame.",
        "**{name}** is here!",
    ]

    FACE_UNKNOWN_TEMPLATES: list[str] = [
        "I see someone I don't recognize. 🤔 Who is this?",
        "There's a face I haven't seen before. Can you tell me who this is?",
        "I detected an unknown person. What's their name?",
        "New face detected! Who should I remember this as?",
    ]

    EMOTION_TEMPLATES: dict[str, list[str]] = {
        "happy": [
            "{name} looks **happy**! 😊",
            "{name} seems to be in a great mood! 😄",
            "I sense happiness from {name}. 🌟",
        ],
        "sad": [
            "{name} looks **sad**. 😢",
            "{name} seems a bit down. 💙",
            "I notice {name} might be feeling sad.",
        ],
        "angry": [
            "{name} appears **angry**. 😠",
            "{name} looks upset. 🔥",
            "I sense some frustration from {name}.",
        ],
        "neutral": [
            "{name} has a **neutral** expression.",
            "{name} looks calm and composed. 😐",
            "{name}'s expression is neutral.",
        ],
        "surprise": [
            "{name} looks **surprised**! 😲",
            "{name} seems caught off guard!",
            "I see surprise on {name}'s face! 😮",
        ],
        "fear": [
            "{name} looks **worried**. 😰",
            "{name} seems a bit anxious.",
            "I notice concern on {name}'s face.",
        ],
        "disgust": [
            "{name} looks **displeased**. 😒",
            "{name} doesn't seem happy about something.",
        ],
    }

    SIGN_TEMPLATES: dict[str, list[str]] = {
        "hello":      ["✋ I see a **Hello** sign! Hi there!", "Someone is waving hello! 👋"],
        "yes":        ["👍 That's a **Yes** / thumbs up!", "I see a **Yes** gesture!"],
        "no":         ["✌️ I see a **No** gesture.", "That looks like a **No** sign."],
        "thank_you":  ["🙏 I see a **Thank You** gesture!", "Someone is signing **Thank You**!"],
        "help":       ["🆘 I see a **Help** sign!", "Someone is signaling for **Help**!"],
        "i_love_you": ["🤟 I see the **I Love You** sign! ❤️", "That's the **I Love You** gesture! 💜"],
    }

    NO_DETECTION_TEMPLATES: list[str] = [
        "I don't see anything notable right now. Try adjusting the camera.",
        "The frame seems clear. Nothing detected at the moment.",
        "🔍 No detections yet. Make sure the camera has a clear view.",
        "Hmm, I can't detect anything right now. Try moving into frame.",
    ]

    # ════════════════════════════════════════════
    # Main Generation
    # ════════════════════════════════════════════

    def generate(
        self,
        objects: list[dict] | None = None,
        faces: dict | list | None = None,
        emotions: list[dict] | None = None,
        signs: list[dict] | None = None,
    ) -> str:
        """
        Generate a human-readable response from detection results.

        Resolves faces, objects, emotions, and signs into a cohesive
        natural-language sentence. Deduplicates near-identical messages.
        """
        parts: list[str] = []
        has_detections = False

        # Make mutable copies so we can consume emotions per-face
        emotions_remaining = list(emotions) if emotions else []

        # ── Faces ──
        face_list, unknown_list = self._extract_face_lists(faces)

        for face in face_list:
            has_detections = True
            if face.get("is_unknown"):
                parts.append(random.choice(self.FACE_UNKNOWN_TEMPLATES))
            else:
                name = face.get("name", "Someone")
                parts.append(
                    random.choice(self.FACE_KNOWN_TEMPLATES).format(name=name)
                )
                # Pair emotions with this named face
                if emotions_remaining:
                    emo = emotions_remaining.pop(0)
                    parts.append(self._format_emotion(emo, name))

        # ── Remaining unpaired emotions ──
        for emo in emotions_remaining:
            has_detections = True
            parts.append(self._format_emotion(emo, "The person"))

        # ── Objects ──
        if objects:
            has_detections = True
            parts.append(self._format_objects(objects))

        # ── Signs ──
        if signs:
            for sign_det in signs:
                has_detections = True
                sign_name = sign_det.get("sign", "")
                templates = self.SIGN_TEMPLATES.get(
                    sign_name, [f"I see a **{sign_name}** gesture."]
                )
                parts.append(random.choice(templates))

        # ── Nothing detected ──
        if not has_detections:
            return random.choice(self.NO_DETECTION_TEMPLATES)

        message = " ".join(parts)

        # ── Dedup within time window ──
        now = time.time()
        if (
            message == self._last_message
            and (now - self._last_message_time) < DEDUP_WINDOW_SECONDS
        ):
            return ""  # empty string = frontend should skip this message

        self._last_message = message
        self._last_message_time = now
        return message

    # ════════════════════════════════════════════
    # Learning Response
    # ════════════════════════════════════════════

    def generate_learning_response(self, item_type: str, label: str) -> str:
        """Generate a confirmation message after learning a new item."""
        if item_type == "face":
            responses = [
                f"Got it! I'll remember **{label}** from now on. 🧠✨",
                f"Thanks! I've learned that this is **{label}**. I'll recognize them next time! 👤",
                f"**{label}** has been saved to my memory. Nice to meet them! 🌙",
                f"Noted! **{label}** is now in my face database. 📸",
            ]
        else:
            responses = [
                f"Thanks! I've learned that this is a **{label}**. 📦",
                f"Got it! **{label}** has been added to my knowledge. 🧠",
                f"I'll remember **{label}** for next time! ✨",
                f"**{label}** saved! My knowledge keeps growing. 🌙",
            ]
        return random.choice(responses)

    # ════════════════════════════════════════════
    # Greeting
    # ════════════════════════════════════════════

    def generate_greeting(self) -> str:
        """Generate a time-aware greeting message."""
        hour = datetime.now().hour
        if hour < 12:
            period = "Good morning"
        elif hour < 17:
            period = "Good afternoon"
        else:
            period = "Good evening"

        return f"{period}! 🌙 LUNA is ready to assist you."

    # ════════════════════════════════════════════
    # Formatting Helpers
    # ════════════════════════════════════════════

    def _format_objects(self, objects: list[dict]) -> str:
        """Format object detections into a readable string."""
        if len(objects) == 1:
            obj = objects[0]
            article = self._get_article(obj["label"])
            return random.choice(self.OBJECT_TEMPLATES).format(
                article=article,
                label=obj["label"],
                confidence=int(obj.get("confidence", 0) * 100),
            )

        if len(objects) <= 5:
            items = ", ".join(f"**{o['label']}**" for o in objects)
            return random.choice(self.MULTI_OBJECT_TEMPLATES).format(items=items)

        # Many objects — group + count
        counts: dict[str, int] = {}
        for obj in objects:
            counts[obj["label"]] = counts.get(obj["label"], 0) + 1

        items = ", ".join(
            f"{cnt}× **{label}**" if cnt > 1 else f"**{label}**"
            for label, cnt in counts.items()
        )
        return random.choice(self.MULTI_OBJECT_TEMPLATES).format(items=items)

    def _format_emotion(self, emo: dict, name: str) -> str:
        """Pick an emotion template and fill in the name."""
        emotion = emo.get("emotion", "neutral")
        templates = self.EMOTION_TEMPLATES.get(
            emotion, self.EMOTION_TEMPLATES["neutral"]
        )
        return random.choice(templates).format(name=name)

    @staticmethod
    def _extract_face_lists(faces) -> tuple[list[dict], list[dict]]:
        """Normalise the *faces* argument into face_list and unknown_list."""
        if faces is None:
            return [], []
        if isinstance(faces, dict):
            return faces.get("faces", []), faces.get("unknown_faces", [])
        if isinstance(faces, list):
            return faces, []
        return [], []

    @staticmethod
    def _get_article(word: str) -> str:
        """Return 'a' or 'an' based on the first letter."""
        return "an" if word and word[0].lower() in "aeiou" else "a"
