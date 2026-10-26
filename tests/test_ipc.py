import asyncio

import pytest

from bare_stow_host import WorkerError, wrap
from pipe import Worker, pair


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 5))


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

        with pytest.raises(WorkerError, match="boom"):
            await ipc.ready

        await closing

    run(body())


def test_rejects_ready_when_the_worker_exits_before_ready():
    async def body():
        left, right = pair()
        ipc = wrap(left)
        worker = Worker(right)
        closing = asyncio.ensure_future(close_on_terminate(worker))

        await worker.send("exit", code=1)

        with pytest.raises(WorkerError, match="exited with code 1 before ready"):
            await ipc.ready

        await closing

    run(body())


def test_rejects_ready_when_the_transport_closes_before_ready():
    async def body():
        left, _ = pair()
        ipc = wrap(left)

        await left.close()

        with pytest.raises(WorkerError, match="closed before ready"):
            await ipc.ready

    run(body())


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

        await worker.close()

        return received, replied, await ipc.read()

    assert run(body()) == (b"payload", b"reply", b"")


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


def test_wait_returns_when_the_transport_closes():
    async def body():
        left, _ = pair()
        ipc = wrap(left)

        await left.close()

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

        with pytest.raises(WorkerError, match="timed out before ready"):
            await ipc.ready

        await closing

    run(body())


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

        with pytest.raises(WorkerError, match="closed before ready"):
            await ipc.ready

        return await ipc.terminate()

    assert run(body()) is None


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

    assert error.message == "Worker exited with code 3"
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

    assert error.message == "Worker was killed by SIGKILL"
    assert error.signal_code == "SIGKILL"


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
