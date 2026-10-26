import asyncio

from bare_stow_host.protocol import (
    CONTROL,
    USER,
    FrameDecoder,
    decode_control,
    encode_control,
    encode_frame,
)


class End:
    def __init__(self):
        self._queue = asyncio.Queue()
        self.peer = None
        self.closed = False

    async def read(self):
        data = await self._queue.get()

        if not data:
            self._queue.put_nowait(b"")

        return data

    async def write(self, data):
        if self.closed:
            raise BrokenPipeError("End is closed")

        self.peer._queue.put_nowait(bytes(data))

    async def close(self):
        for end in (self, self.peer):
            if not end.closed:
                end.closed = True
                end._queue.put_nowait(b"")


def pair():
    a, b = End(), End()
    a.peer, b.peer = b, a
    return a, b


class Worker:
    def __init__(self, end):
        self.end = end
        self._decoder = FrameDecoder()
        self._frames = asyncio.Queue()
        self._reader = asyncio.ensure_future(self._read())

    async def _read(self):
        while data := await self.end.read():
            for frame in self._decoder.feed(data):
                self._frames.put_nowait(frame)

    async def send(self, type, **payload):
        await self.end.write(encode_control(type, **payload))

    async def write(self, data):
        await self.end.write(encode_frame(USER, data))

    async def receive(self, type):
        while True:
            tag, payload = await self._frames.get()

            if tag == CONTROL and decode_control(payload).get("type") == type:
                return decode_control(payload)

    async def data(self):
        while True:
            tag, payload = await self._frames.get()

            if tag == USER:
                return payload

    async def close(self):
        await self.end.close()
        await self._reader
