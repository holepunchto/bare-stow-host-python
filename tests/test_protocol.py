import pytest

from bare_stow_host.protocol import (
    CONTROL,
    USER,
    FrameDecoder,
    decode_control,
    encode_control,
    encode_frame,
)

# Written by the JavaScript bare-stow-protocol for the same frames.
VECTORS = [
    (
        encode_control("configure", grace=5000),
        "22000000007b2274797065223a22636f6e666967757265222c226772616365223a353030307d",
    ),
    (encode_control("ready"), "11000000007b2274797065223a227265616479227d"),
    (encode_frame(USER, b"hello"), "060000000168656c6c6f"),
]


@pytest.mark.parametrize(("frame", "expected"), VECTORS)
def test_encodes_frames_like_the_javascript_implementation(frame, expected):
    assert frame == bytes.fromhex(expected)


def test_decodes_frames_split_across_chunks():
    decoder = FrameDecoder()
    data = encode_frame(USER, b"hello") + encode_control("ready")

    frames = []

    for i in range(len(data)):
        frames += decoder.feed(data[i : i + 1])

    assert frames == [(USER, b"hello"), (CONTROL, b'{"type":"ready"}')]


def test_decodes_several_frames_from_one_chunk():
    decoder = FrameDecoder()
    data = encode_frame(USER, b"a") + encode_frame(USER, b"") + encode_frame(USER, b"b")

    assert decoder.feed(data) == [(USER, b"a"), (USER, b""), (USER, b"b")]


def test_skips_empty_frames():
    decoder = FrameDecoder()

    assert decoder.feed(b"\x00\x00\x00\x00" + encode_frame(USER, b"a")) == [
        (USER, b"a")
    ]


def test_ignores_invalid_control_payloads():
    assert decode_control(b"not json") is None
    assert decode_control(b"[1, 2]") is None
    assert decode_control(b'{"type":"ready"}') == {"type": "ready"}
