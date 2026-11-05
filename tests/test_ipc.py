import asyncio

import pytest

from bare_stow_host import WorkerError, wrap
from bare_stow_host.protocol import encode_control
from pipe import Worker, pair


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 5))


async def settle():
    for _ in range(10):
        await asyncio.sleep(0)


class BrokenTransport:
    def __init__(self, *, read_error=None, write_error=None):
        self._read_error = read_error
        self.write_error = write_error
        self._reads = asyncio.Queue()

    async def read(self):
        if self._read_error is not None:
            raise self._read_error

        return await self._reads.get()

    async def write(self, data):
        if self.write_error is not None:
            raise self.write_error

    def feed(self, data):
        self._reads.put_nowait(data)

    def close(self):
        self._reads.put_nowait(b"")


async def close_on_terminate(worker):
    await worker.receive("terminate")
    await worker.close()


def test_resolves_ready_when_the_worker_signals():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        await worker.send("exit", code=0)
        await worker.close()
        await ipc.wait()

    run(body())


def test_sends_the_grace_to_the_worker():
    async def body():
        left, right = pair()
        ipc = wrap(left, grace=2.5)
        worker = Worker(right)

        message = await worker.receive("configure")

        await worker.close()
        ipc.exited(0)
        await ipc.wait()

        return message

    assert run(body()) == {"type": "configure", "grace": 2500}


def test_rejects_ready_when_the_worker_errors_before_ready():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)
        closing = asyncio.ensure_future(close_on_terminate(worker))

        await worker.send("error", message="boom")

        with pytest.raises(WorkerError, match="boom") as err:
            await ipc.ready

        await closing

        return err.value

    assert run(body()).code == "WORKER_ERROR"


def test_rejects_ready_when_the_worker_exits_before_ready():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)
        closing = asyncio.ensure_future(close_on_terminate(worker))

        await worker.send("exit", code=1)

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await closing

        return err.value

    error = run(body())

    assert error.code == "WORKER_EXITED"
    assert error.exit_code == 1


def test_rejects_ready_when_closed_before_ready():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)
        closing = asyncio.ensure_future(close_on_terminate(worker))

        await ipc.close()
        await closing

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        return err.value

    assert run(body()).code == "WORKER_CLOSED"


def test_ignores_exit_after_ready():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        await worker.send("exit", code=1)
        await worker.close()
        await ipc.wait()

        return await ipc.ready

    assert run(body()) is None


def test_reports_the_exit_code():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)

        await worker.send("ready")
        await worker.send("exit", code=7)
        await worker.close()

        return await ipc.wait(), ipc.exit_code

    assert run(body()) == (7, 7)


def test_carries_user_data():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)

        await worker.write(b"payload")
        received = await ipc.read()

        await ipc.write(b"reply")
        replied = await worker.data()

        await worker.send("exit", code=0)
        await worker.close()

        with pytest.raises(WorkerError) as err:
            await ipc.write(b"late")

        return received, replied, await ipc.read(), err.value.code

    assert run(body()) == (b"payload", b"reply", b"", "WORKER_CLOSED")


def test_terminate_resolves_to_the_exit_code():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        async def exit_on_terminate():
            await worker.receive("terminate")
            await worker.send("exit", code=0)
            await worker.close()

        exiting = asyncio.ensure_future(exit_on_terminate())
        code = await ipc.terminate()
        await exiting

        return code

    assert run(body()) == 0


def test_close_sends_terminate_to_the_worker():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)
        closing = asyncio.ensure_future(close_on_terminate(worker))

        await ipc.close()
        await closing

    run(body())


def test_wait_returns_once_a_held_end_is_released():
    async def body():
        left, _ = pair()
        ipc = wrap(left)

        await left.close()
        ipc.exited(1)

        with pytest.raises(WorkerError):
            await ipc.ready

        return await ipc.wait()

    assert run(body()) is None


def test_rejects_ready_and_terminates_the_worker_on_timeout():
    async def body():
        left, right = pair()
        ipc = wrap(left, timeout=0.01)
        worker = Worker(right)
        closing = asyncio.ensure_future(close_on_terminate(worker))

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await closing

        return err.value

    assert run(body()).code == "WORKER_TIMEOUT"


def test_terminates_the_worker_when_ready_is_cancelled():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)
        closing = asyncio.ensure_future(close_on_terminate(worker))

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ipc.ready, 0.01)

        await closing

        return await ipc.wait()

    assert run(body()) is None


def test_terminate_kills_the_worker_after_the_grace_period():
    async def body():
        left, right = pair()
        ipc = wrap(left, grace=0.01)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        return await ipc.terminate(), left.closed

    assert run(body()) == (None, True)


def test_terminate_calls_kill_after_the_grace_period():
    async def body():
        left, right = pair()
        killed = []

        def kill():
            killed.append(True)
            return right.close()

        ipc = wrap(left, grace=0.01, kill=kill)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        return await ipc.terminate(), killed

    assert run(body()) == (None, [True])


def test_close_kills_the_worker_after_the_grace_period():
    async def body():
        left, _ = pair()
        ipc = wrap(left, grace=0.01)

        await ipc.close()

        return left.closed

    assert run(body()) is True


def test_terminate_resolves_once_the_transport_has_closed():
    async def body():
        left, _ = pair()
        ipc = wrap(left)

        await left.close()
        ipc.exited(1)

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        return err.value.code, await ipc.terminate()

    assert run(body()) == ("WORKER_EXITED", None)


def test_exited_before_ready_rejects_ready_with_the_exit_code():
    async def body():
        left, _ = pair()
        ipc = wrap(left, grace=0.01)

        ipc.exited(3)

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await ipc.close()

        return err.value

    error = run(body())

    assert error.code == "WORKER_EXITED"
    assert error.exit_code == 3


def test_exited_after_ready_raises_with_the_signal():
    async def body():
        left, right = pair()
        ipc = wrap(left, grace=0.01)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        ipc.exited(None, "SIGKILL")

        with pytest.raises(WorkerError) as err:
            await ipc.read()

        await ipc.close()

        return err.value

    error = run(body())

    assert error.code == "WORKER_KILLED"
    assert error.exit_code is None
    assert error.signal_code == "SIGKILL"


def test_exited_with_a_code_and_a_signal_carries_both():
    async def body():
        left, _ = pair()
        ipc = wrap(left, grace=0.01)

        ipc.exited(1, "SIGTERM")

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await ipc.close()

        return err.value

    error = run(body())

    assert error.code == "WORKER_KILLED"
    assert error.exit_code == 1
    assert error.signal_code == "SIGTERM"


def test_kills_a_worker_after_grace_when_a_control_frame_fails():
    cause = OSError("unreachable")

    async def body():
        killed = []
        transport = BrokenTransport(write_error=cause)

        def kill():
            killed.append(True)
            transport.close()

        ipc = wrap(transport, grace=0.01, kill=kill)

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await ipc.close()

        return killed, err.value

    killed, error = run(body())

    assert killed == [True]
    assert error.code == "TRANSPORT_ERROR"
    assert error.__cause__ is cause


def test_kills_a_worker_after_grace_when_writing_fails():
    cause = OSError("unreachable")

    async def body():
        killed = []
        transport = BrokenTransport()

        def kill():
            killed.append(True)
            transport.close()

        ipc = wrap(transport, grace=0.01, kill=kill)

        await settle()

        transport.write_error = cause

        with pytest.raises(WorkerError) as write:
            await ipc.write(b"data")

        with pytest.raises(WorkerError) as ready:
            await ipc.ready

        await ipc.close()

        return killed, write.value, ready.value

    killed, write, ready = run(body())

    assert killed == [True]
    assert write.code == "TRANSPORT_ERROR"
    assert write.__cause__ is cause
    assert ready is write


def test_ignores_a_transport_failure_after_terminate_is_sent():
    async def body():
        transport = BrokenTransport()
        ipc = wrap(transport, grace=0.01)

        await settle()

        transport.write_error = OSError("unreachable")

        return await ipc.terminate(), await ipc.read()

    assert run(body()) == (None, b"")


def test_fails_a_crashed_worker_with_its_exit_rather_than_a_transport_failure():
    cause = OSError("unreachable")

    async def body():
        ipc = wrap(BrokenTransport(read_error=cause))

        await settle()

        ipc.exited(None, "SIGSEGV")

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await ipc.wait()

        return err.value

    error = run(body())

    assert error.code == "WORKER_KILLED"
    assert error.__cause__.code == "TRANSPORT_ERROR"
    assert error.__cause__.__cause__ is cause


def test_kills_a_worker_whose_transport_fails_and_that_does_not_exit():
    cause = OSError("unreachable")

    async def body():
        killed = []
        ipc = wrap(
            BrokenTransport(read_error=cause),
            grace=0.01,
            kill=lambda: killed.append(True),
        )

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        ipc.exited(1)

        with pytest.raises(WorkerError) as read:
            await ipc.read()

        await ipc.wait()

        return killed, err.value, read.value

    killed, error, read = run(body())

    assert killed == [True]
    assert error.code == "TRANSPORT_ERROR"
    assert error.__cause__ is cause
    assert read is error


def test_kills_a_worker_whose_transport_fails_and_then_ends():
    cause = OSError("unreachable")

    async def body():
        killed = []
        transport = BrokenTransport()
        ipc = wrap(
            transport,
            grace=0.01,
            kill=lambda: killed.append(True),
        )

        await settle()

        transport.write_error = cause

        with pytest.raises(WorkerError):
            await ipc.write(b"data")

        transport.close()

        with pytest.raises(WorkerError) as ready:
            await ipc.ready

        with pytest.raises(WorkerError) as read:
            await ipc.read()

        await ipc.wait()

        return killed, ready.value, read.value

    killed, ready, read = run(body())

    assert killed == [True]
    assert ready.code == "TRANSPORT_ERROR"
    assert ready.__cause__ is cause
    assert read is ready


def test_kills_a_worker_that_exits_after_its_transport_fails():
    async def body():
        killed = []
        transport = BrokenTransport()
        ipc = wrap(
            transport,
            grace=0.01,
            kill=lambda: killed.append(True),
        )

        await settle()

        transport.write_error = OSError("unreachable")

        with pytest.raises(WorkerError):
            await ipc.write(b"data")

        transport.feed(encode_control("ready"))
        transport.feed(encode_control("exit", code=0))
        transport.close()

        code = await ipc.wait()

        await asyncio.sleep(0.05)

        return killed, code

    assert run(body()) == ([True], 0)


def test_treats_a_reset_transport_as_closed():
    async def body():
        ipc = wrap(
            BrokenTransport(read_error=ConnectionResetError("reset")),
        )

        await settle()

        ipc.exited(1)

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await ipc.wait()

        return err.value

    error = run(body())

    assert error.code == "WORKER_EXITED"
    assert error.__cause__ is None


def test_exited_is_ignored_after_an_exit_frame():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        await worker.send("exit", code=0)
        await worker.close()
        await ipc.wait()

        ipc.exited(0)

        return await ipc.read()

    assert run(body()) == b""


def test_exited_is_ignored_while_terminating():
    async def body():
        left, right = pair()
        ipc = wrap(left, grace=0.01)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        terminated = ipc.terminate()

        ipc.exited(None, "SIGKILL")

        return await terminated, await ipc.read()

    assert run(body()) == (None, b"")


def test_holds_the_end_of_a_worker_that_crashes_until_its_exit_is_reported():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        reading = asyncio.ensure_future(ipc.read())

        await worker.close()
        await settle()

        held = not reading.done()

        ipc.exited(None, "SIGSEGV")

        with pytest.raises(WorkerError) as err:
            await reading

        await ipc.wait()

        return held, err.value

    held, error = run(body())

    assert held
    assert error.code == "WORKER_KILLED"
    assert error.signal_code == "SIGSEGV"


def test_holds_the_end_of_a_worker_that_crashes_before_ready():
    async def body():
        left, _ = pair()
        ipc = wrap(left)

        await left.close()
        await settle()

        held = not ipc.ready.done()

        ipc.exited(1)

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await ipc.wait()

        return held, err.value

    held, error = run(body())

    assert held
    assert error.code == "WORKER_EXITED"
    assert error.exit_code == 1


def test_fails_a_crashed_worker_with_its_exit_rather_than_a_timeout():
    async def body():
        left, _ = pair()
        ipc = wrap(left, timeout=0.05)

        await left.close()
        await asyncio.sleep(0.1)

        ipc.exited(1)

        with pytest.raises(WorkerError) as err:
            await ipc.ready

        await ipc.wait()

        return err.value

    assert run(body()).code == "WORKER_EXITED"


def test_exited_applies_to_a_close_after_the_transport_ended():
    async def body():
        left, right = pair()
        killed = []
        ipc = wrap(left, grace=0.01, kill=lambda: killed.append(True))
        worker = Worker(right)

        await worker.send("ready")
        await ipc.ready

        await worker.close()
        await settle()

        closing = ipc.close()

        await asyncio.sleep(0.05)

        ipc.exited(1)

        with pytest.raises(WorkerError) as err:
            await ipc.read()

        await closing

        return killed, err.value

    killed, error = run(body())

    assert killed == []
    assert error.code == "WORKER_EXITED"
    assert error.exit_code == 1


def test_does_not_hold_the_end_of_a_worker_that_exits_on_its_own():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)

        await worker.send("ready")
        await worker.send("exit", code=0)
        await worker.close()

        return await ipc.read(), await ipc.wait()

    assert run(body()) == (b"", 0)
