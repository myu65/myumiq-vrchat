"""Bounded text-only bridge from model worker to the existing audio executive."""

import json
import re
from collections import deque
from threading import Lock


def reply_prefix(raw):
    """Decode only a leading reply string, including split JSON escapes.

    Other property orders simply wait for the validated final object. Nothing
    in topic, quoted history or model control fields can become early speech.
    """
    match = re.match(r'^\s*\{\s*"reply"\s*:\s*"', raw)
    if not match:
        return ""
    start = index = valid = match.end()
    while index < len(raw):
        char = raw[index]
        if char == '"':
            break
        if char == "\\":
            if index + 1 >= len(raw):
                break
            size = 6 if raw[index + 1] == "u" else 2
            if index + size > len(raw):
                break
            index += size
        else:
            index += 1
        valid = index
    decoded = json.loads('"' + raw[start:valid] + '"')
    # Wait for the low surrogate rather than passing a lone surrogate to TTS.
    return decoded[:-1] if decoded and 0xD800 <= ord(decoded[-1]) <= 0xDBFF else decoded


class SentenceReply:
    def __init__(self):
        self.guard = Lock()
        self.raw = ""
        self.emitted = ""
        self.ready = deque()
        self.submitted = ""

    def feed(self, delta):
        self.raw += delta
        if len(self.raw) > 65536:
            raise ValueError("dialogue stream too large")
        prefix = reply_prefix(self.raw)
        if len(prefix) > 180:
            raise ValueError("dialogue reply too long")
        rest = prefix[len(self.emitted) :]
        for match in re.finditer(r"[。！？!?]", rest):
            sentence = rest[: match.end()]
            if re.search(r"[ぁ-ゖァ-ヺ]", sentence):
                self._append(sentence)
                # Recompute the remaining prefix to avoid overlapping chunks.
                self.feed("")
                return

    def _append(self, text):
        with self.guard:
            if len(self.ready) >= 16:
                raise ValueError("too many pending speech segments")
            self.ready.append(text)
            self.emitted += text

    def finish(self, reply):
        if not reply.startswith(self.emitted):
            raise ValueError("final dialogue differs from streamed reply")
        remaining = reply[len(self.emitted) :]
        if remaining:
            self._append(remaining)

    def peek(self):
        with self.guard:
            return self.ready[0] if self.ready else None

    def acknowledge(self):
        with self.guard:
            self.submitted += self.ready.popleft()
