import json
import struct

CONTROL = 0x00
"""The tag of a control frame."""

USER = 0x01
"""The tag of a user data frame."""

DEFAULT_GRACE = 5.0
"""The default number of seconds a terminating worker gets to stop and exit."""

_HEADER = struct.Struct("<IB")
_LENGTH = struct.Struct("<I")


def encode_frame(tag, data):
    return _HEADER.pack(len(data) + 1, tag) + bytes(data)


def encode_control(type, **payload):
    message = json.dumps({"type": type, **payload}, separators=(",", ":"))

    return encode_frame(CONTROL, message.encode())


def decode_control(payload):
    try:
        message = json.loads(payload)
    except ValueError:
        return None

    return message if isinstance(message, dict) else None


class FrameDecoder:
    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data):
        self._buffer += data

        frames = []
        offset = 0

        with memoryview(self._buffer) as view:
            while len(view) - offset >= _LENGTH.size:
                (length,) = _LENGTH.unpack_from(view, offset)
                start = offset + _LENGTH.size
                end = start + length

                if len(view) < end:
                    break

                if length > 0:
                    frames.append((view[start], bytes(view[start + 1 : end])))

                offset = end

        del self._buffer[:offset]

        return frames
