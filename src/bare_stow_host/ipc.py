import asyncio
import inspect
from contextlib import suppress

from .protocol import (
    CONTROL,
    DEFAULT_GRACE,
    USER,
    FrameDecoder,
    decode_control,
    encode_control,
    encode_frame,
)

_END = object()

_PEER_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class WorkerError(Exception):
    """Raised for faults reported by or observed in the worker.

    ``code`` tells the failures apart: ``WORKER_ERROR`` for a fault the worker
    reported, ``WORKER_EXITED`` for a worker that exited, ``WORKER_KILLED`` for
    a worker killed by a signal, ``WORKER_CLOSED`` for a transport that closed
    before ready, ``WORKER_TIMEOUT`` for a worker that timed out before ready,
    and ``TRANSPORT_ERROR`` for a transport that failed, whose error is chained
    as ``__cause__``. A transport that is reset or broken by the worker going
    away counts as closing rather than failing.

    ``message`` and ``stack`` carry the error reported by the worker, while
    ``exit_code`` and ``signal_code`` carry the reason a worker that exited
    did so.
    """

    def __init__(
        self, message, *, code=None, stack=None, exit_code=None, signal_code=None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.stack = stack
        self.exit_code = exit_code
        self.signal_code = signal_code


def _transport_error(cause):
    error = WorkerError(str(cause), code="TRANSPORT_ERROR")
    error.__cause__ = cause
    return error


def wrap(transport, **opts):
    """Wrap the host side of a stowed bundle's ``transport`` in an ``IPC``.

    See ``IPC`` for the options.
    """
    return IPC(transport, **opts)


class IPC:
    """The host side of a stowed bundle's lifecycle.

    ``transport`` has an async ``read()`` returning ``b""`` once it has ended,
    an async ``write(data)``, and ``close()``. The ``grace`` is sent to the
    worker straight away. ``kill`` stops a worker that ignores ``terminate``
    for ``grace`` seconds and defaults to ``transport.close()``. ``timeout``
    is the number of seconds to wait for the worker to signal ready before
    failing ``ready`` and terminating it, or ``None`` to wait indefinitely.

    Whoever wraps the transport must call ``exited()`` once the worker exits.
    The end of a transport that ends without the worker saying why is held
    back until then, so that the handle fails with the reason, and the worker
    can no longer time out before ready. A transport that fails instead waits
    for the exit for ``grace`` seconds, failing with the exit and the failure
    as its ``__cause__``, or killing the worker and failing with
    ``TRANSPORT_ERROR`` if it hasn't exited by then.

    Iterating it with ``async for`` yields the user data from the worker until
    the transport ends, and leaving ``async with`` closes it.
    """

    def __init__(
        self,
        transport,
        *,
        kill=None,
        timeout=None,
        grace=DEFAULT_GRACE,
    ):
        loop = asyncio.get_running_loop()

        self._transport = transport
        self._kill = kill or transport.close
        self._grace = grace
        self._decoder = FrameDecoder()
        self._data = asyncio.Queue()
        self._writing = asyncio.Lock()
        self._ready = loop.create_future()
        self._closed = loop.create_future()
        self._exit_code = None
        self._exit_reported = False
        self._error = None
        self._terminating = None
        self._asked_to_terminate = False
        self._ended = False
        self._failure = None
        self._failing = None
        self._tasks = set()

        self._timer = None

        if timeout is not None:
            self._timer = loop.call_later(
                timeout,
                self._settle_ready,
                WorkerError("Worker timed out before ready", code="WORKER_TIMEOUT"),
            )

        self._ready.add_done_callback(self._on_ready_settled)

        self._spawn(self._send(encode_control("configure", grace=round(grace * 1000))))
        self._spawn(self._read_frames())

    @property
    def ready(self):
        """A future that resolves once the worker has signaled ready.

        It raises ``WorkerError`` if the worker errors, exits, closes, or times
        out before then, in which case the worker is terminated.
        """
        return self._ready

    @property
    def exit_code(self):
        """The exit code the worker reported, or ``None``."""
        return self._exit_code

    async def read(self):
        """Read the next chunk of user data from the worker, or ``b""`` once
        the transport has ended.

        Raises ``WorkerError`` once the worker has reported a fault or exited
        without saying why.
        """
        item = await self._data.get()

        if isinstance(item, bytes):
            return item

        self._data.put_nowait(item)

        if item is _END:
            return b""

        raise item

    def __aiter__(self):
        return self._chunks()

    async def _chunks(self):
        while chunk := await self.read():
            yield chunk

    async def write(self, data):
        """Write ``data`` to the worker as user data.

        Raises a ``WorkerError`` with code ``WORKER_CLOSED`` if the transport
        has closed, or ``TRANSPORT_ERROR`` if it fails.
        """
        await self._write(encode_frame(USER, data))

    async def wait(self):
        """Wait for the transport to close and return the exit code the worker
        reported, or ``None``."""
        await asyncio.shield(self._closed)

        return self._exit_code

    def terminate(self):
        """Ask the worker to terminate and return a future of the exit code it
        reported, or ``None`` if it closed without reporting one.

        Terminating starts as soon as it is called. The transport is waited
        for to close, calling ``kill`` if the worker hasn't exited within
        ``grace``.
        """
        return asyncio.ensure_future(self._exit_code_after(self.close()))

    def close(self):
        """Like ``terminate()``, without the exit code."""
        if self._terminating is None:
            self._asked_to_terminate = not self._ended
            self._terminating = asyncio.ensure_future(self._terminate())

        return asyncio.shield(self._terminating)

    def exited(self, code, signal=None):
        """Report that the worker has exited, as observed by the transport,
        such as the process of a sidecar. Whoever wraps the transport must call
        it once the worker exits.

        If the worker exited without saying why and was not asked to, the
        handle fails with a ``WORKER_EXITED`` or ``WORKER_KILLED``
        ``WorkerError`` carrying ``exit_code`` and ``signal_code``.
        """
        if self._exit_reported or self._asked_to_terminate or self._error is not None:
            return

        if signal is None:
            error = WorkerError(
                f"Worker exited with code {code}", code="WORKER_EXITED", exit_code=code
            )
        else:
            error = WorkerError(
                f"Worker was killed by {signal}",
                code="WORKER_KILLED",
                exit_code=code,
                signal_code=signal,
            )

        if self._failure is not None:
            self._failing.cancel()
            error.__cause__ = self._failure

        self._fail(error)
        self._release()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def _spawn(self, coro):
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _write(self, frame):
        async with self._writing:
            try:
                await self._transport.write(frame)
            except Exception as err:
                raise self._transport_failed(err) from err

    async def _send(self, frame):
        with suppress(WorkerError):
            await self._write(frame)

    async def _read_frames(self):
        try:
            while data := await self._transport.read():
                for tag, payload in self._decoder.feed(data):
                    self._on_frame(tag, payload)
        except Exception as err:
            self._transport_failed(err)

        self._ended = True

        # A worker whose transport ends without it saying why has crashed, so
        # the end is held back until its exit is reported for readers to fail
        # with.
        if not self._crashed():
            self._on_close()
        elif self._timer is not None:
            self._timer.cancel()

    # A peer that goes away resets or breaks the transport rather than ending
    # it, which is no different from it closing. Other failures of the
    # transport of a worker that may have crashed wait for its exit as an end
    # does, killing it if it hasn't exited within the grace period.
    def _transport_failed(self, err):
        if isinstance(err, _PEER_GONE) or self._ending():
            error = WorkerError("Worker closed", code="WORKER_CLOSED")
            error.__cause__ = err
            return error

        failure = _transport_error(err)

        if self._failure is not None:
            return failure

        if self._crashed():
            if self._timer is not None:
                self._timer.cancel()

            self._failure = failure
            self._failing = asyncio.get_running_loop().call_later(
                self._grace, self._give_up
            )
        else:
            self._fail(failure)

        return failure

    def _give_up(self):
        self._call_kill()
        self._fail(self._failure)
        self._release()

    def _ending(self):
        return self._ended or self._exit_reported or self._asked_to_terminate

    def _release(self):
        if self._ended and not self._closed.done():
            self._on_close()

    def _crashed(self):
        return (
            not self._exit_reported
            and not self._asked_to_terminate
            and self._error is None
        )

    def _on_frame(self, tag, payload):
        if tag == USER:
            self._data.put_nowait(payload)
            return

        if tag != CONTROL:
            return

        message = decode_control(payload)

        if message is None:
            return

        kind = message.get("type")

        if kind == "ready":
            self._settle_ready()
        elif kind == "exit":
            self._on_exit(message.get("code"))
        elif kind == "error":
            error = WorkerError(
                message.get("message", ""),
                code="WORKER_ERROR",
                stack=message.get("stack"),
            )
            self._fail(error)

    def _on_exit(self, code):
        self._exit_reported = True
        self._exit_code = code

        # An error frame precedes the exit frame of a worker that faulted.
        if self._error is None:
            self._settle_ready(
                WorkerError(
                    f"Worker exited with code {code} before ready",
                    code="WORKER_EXITED",
                    exit_code=code,
                )
            )

    def _on_close(self):
        if not self._closed.done():
            self._closed.set_result(None)

        self._settle_ready(
            WorkerError("Worker closed before ready", code="WORKER_CLOSED")
        )
        self._data.put_nowait(_END)

    def _fail(self, error):
        if self._error is not None:
            return

        self._error = error
        self._settle_ready(error)
        self._data.put_nowait(error)
        self._spawn(self.close())

    def _settle_ready(self, error=None):
        if self._ready.done():
            return

        if error is None:
            self._ready.set_result(None)
        else:
            self._ready.set_exception(error)

    def _on_ready_settled(self, ready):
        if self._timer is not None:
            self._timer.cancel()

        if ready.cancelled() or ready.exception() is not None:
            self._spawn(self.close())

    async def _terminate(self):
        if not self._asked_to_terminate:
            await asyncio.shield(self._closed)
            return

        timer = asyncio.get_running_loop().call_later(self._grace, self._call_kill)

        try:
            await self._send(encode_control("terminate"))
            await asyncio.shield(self._closed)
        finally:
            timer.cancel()

    async def _exit_code_after(self, closing):
        await closing

        return self._exit_code

    def _call_kill(self):
        result = self._kill()

        if inspect.isawaitable(result):
            self._spawn(result)
