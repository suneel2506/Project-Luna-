"""
Project LUNA — Natural Language Response Generator
Creates human-like, conversational responses from detection results.
Combines multiple detections into coherent messages.
"""

import random
import logging

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate natural language responses from AI detection results."""

    # ---- Response Templates ----

    GREETING_TEMPLATES = [
        "👋 Hello! Let me analyze what I see...",
        "🔍 Scanning the scene now...",
        "👀 Let me take a look...",
        "🌙 LUNA is observing...",
    ]

    OBJECT_TEMPLATES = [
        "I can see {article} **{label}** {location}.",
        "There's {article} **{label}** in the frame.",
        "I detected {article} **{label}** ({confidence}% confident).",
        "I spot {article} **{label}**!",
    ]

    MULTI_OBJECT_TEMPLATES = [
        "I can see several things: {items}.",
        "In the frame, I notice: {items}.",
        "Here's what I detect: {items}.",
    ]

    FACE_KNOWN_TEMPLATES = [
        "I recognize **{name}**! 👋",
        "Hey, it's **{name}**!",
        "I see **{name}** in the frame.",
        "**{name}** is here!",
    ]

    FACE_UNKNOWN_TEMPLATES = [
        "I see someone I don't recognize. 🤔 Who is this?",
        "There's a face I haven't seen before. Can you tell me who this is?",
        "I detected an unknown person. What's their name?",
        "New face detected! Who should I remember this as?",
    ]

    EMOTION_TEMPLATES = {
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

    SIGN_TEMPLATES = {
        "hello": [
            "✋ I see a **Hello** sign! Hi there!",
            "Someone is waving hello! 👋",
        ],
        "yes": [
            "👍 That's a **Yes** / thumbs up!",
            "I see a **Yes** gesture!",
        ],
        "no": [
            "✌️ I see a **No** gesture.",
            "That looks like a **No** sign.",
        ],
        "thank_you": [
            "🙏 I see a **Thank You** gesture!",
            "Someone is signing **Thank You**!",
        ],
        "help": [
            "🆘 I see a **Help** sign!",
            "Someone is signaling for **Help**!",
        ],
        "i_love_you": [
            "🤟 I see the **I Love You** sign! ❤️",
            "That's the **I Love You** gesture! 💜",
        ],
    }

    NO_DETECTION_TEMPLATES = [
        "I don't see anything notable right now. Try adjusting the camera.",
        "The frame seems clear. Nothing detected at the moment.",
        "🔍 No detections yet. Make sure the camera has a clear view.",
        "Hmm, I can't detect anything right now. Try moving into frame.",
    ]

    def generate(self, objects=None, faces=None, emotions=None, signs=None):
        """
        Generate a natural language response from detection results.

        Args:
            objects: list of detected objects [{label, confidence, bbox}]
            faces: dict with 'faces' and 'unknown_faces' lists
            emotions: list of detected emotions [{emotion, confidence, location}]
            signs: list of detected signs [{sign, confidence, hand}]

        Returns:
            str: Human-readable response message
        """
        parts = []
        has_detections = False

        # ---- Process faces ----
        if faces:
            face_list = faces.get("faces", []) if isinstance(faces, dict) else faces
            unknown_list = faces.get("unknown_faces", []) if isinstance(faces, dict) else []

            for face in face_list:
                has_detections = True
                if face.get("is_unknown"):
                    parts.append(random.choice(self.FACE_UNKNOWN_TEMPLATES))
                else:
                    name = face.get("name", "Someone")
                    parts.append(
                        random.choice(self.FACE_KNOWN_TEMPLATES).format(name=name)
                    )

                    # Add emotion for this face if available
                    if emotions:
                        for emo in emotions:
                            emotion = emo.get("emotion", "neutral")
                            templates = self.EMOTION_TEMPLATES.get(emotion, self.EMOTION_TEMPLATES["neutral"])
                            parts.append(
                                random.choice(templates).format(name=name)
                            )
                        # Clear emotions so they aren't repeated
                        emotions = []

        # ---- Process remaining emotions (no face match) ----
        if emotions:
            for emo in emotions:
                has_detections = True
                emotion = emo.get("emotion", "neutral")
                templates = self.EMOTION_TEMPLATES.get(emotion, self.EMOTION_TEMPLATES["neutral"])
                parts.append(
                    random.choice(templates).format(name="The person")
                )

        # ---- Process objects ----
        if objects:
            has_detections = True
            if len(objects) == 1:
                obj = objects[0]
                article = self._get_article(obj["label"])
                parts.append(
                    random.choice(self.OBJECT_TEMPLATES).format(
                        article=article,
                        label=obj["label"],
                        confidence=int(obj.get("confidence", 0) * 100),
                        location="in the frame"
                    )
                )
            elif len(objects) <= 5:
                items = ", ".join([
                    f"**{obj['label']}**" for obj in objects
                ])
                parts.append(
                    random.choice(self.MULTI_OBJECT_TEMPLATES).format(items=items)
                )
            else:
                # Many objects — summarize
                item_counts = {}
                for obj in objects:
                    item_counts[obj["label"]] = item_counts.get(obj["label"], 0) + 1
                items = ", ".join([
                    f"{count}x **{label}**" if count > 1 else f"**{label}**"
                    for label, count in item_counts.items()
                ])
                parts.append(
                    random.choice(self.MULTI_OBJECT_TEMPLATES).format(items=items)
                )

        # ---- Process signs ----
        if signs:
            for sign_det in signs:
                has_detections = True
                sign = sign_det.get("sign", "")
                templates = self.SIGN_TEMPLATES.get(sign, [f"I see a **{sign}** gesture."])
                parts.append(random.choice(templates))

        # ---- No detections ----
        if not has_detections:
            return random.choice(self.NO_DETECTION_TEMPLATES)

        return " ".join(parts)

    def generate_learning_response(self, item_type, label):
        """
        Generate a response after learning a new item.

        Args:
            item_type: 'face' or 'object'
            label: The learned label

        Returns:
            str: Confirmation message
        """
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

    def generate_greeting(self):
        """Generate a greeting message."""
        return random.choice(self.GREETING_TEMPLATES)

    @staticmethod
    def _get_article(word):
        """Get the appropriate article (a/an) for a word."""
        vowels = "aeiouAEIOU"
        return "an" if word[0] in vowels else "a"
