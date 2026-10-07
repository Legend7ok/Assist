import asyncio
import threading

import pytest

from assist.core.bus import EventBus
from assist.core.events import Event


class Ping(Event):
    n: int


class Pong(Event):
    n: int


class Recorder:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)


@pytest.fixture
async def bus():
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.close()


async def settle() -> None:
    """Let worker tasks handle what is already queued."""
    for _ in range(5):
        await asyncio.sleep(0)


async def test_event_reaches_only_subscribers_of_its_type(bus):
    pings, pongs = Recorder(), Recorder()
    bus.subscribe(Ping, pings)
    bus.subscribe(Pong, pongs)

    bus.publish(Ping(n=1))
    await settle()

    assert pings.events == [Ping(n=1)]
    assert pongs.events == []


async def test_subscriber_to_base_class_receives_every_event(bus):
    everything = Recorder()
    bus.subscribe(Event, everything)

    bus.publish(Ping(n=1))
    bus.publish(Pong(n=2))
    await settle()

    assert everything.events == [Ping(n=1), Pong(n=2)]


async def test_events_arrive_in_publish_order(bus):
    pings = Recorder()
    bus.subscribe(Ping, pings)

    for n in range(50):
        bus.publish(Ping(n=n))
    await settle()

    assert [e.n for e in pings.events] == list(range(50))


async def test_slow_subscriber_does_not_delay_others(bus):
    release = asyncio.Event()

    async def slow(_: Ping) -> None:
        await release.wait()

    fast = Recorder()
    bus.subscribe(Ping, slow)
    bus.subscribe(Ping, fast)

    bus.publish(Ping(n=1))
    await settle()

    assert fast.events == [Ping(n=1)]
    release.set()


async def test_failing_handler_does_not_stop_its_subscriber_or_others(bus):
    handled: list[int] = []

    async def flaky(event: Ping) -> None:
        if event.n == 1:
            raise ValueError("boom")
        handled.append(event.n)

    other = Recorder()
    bus.subscribe(Ping, flaky)
    bus.subscribe(Ping, other)

    bus.publish(Ping(n=1))
    bus.publish(Ping(n=2))
    await settle()

    assert handled == [2]
    assert other.events == [Ping(n=1), Ping(n=2)]


async def test_subscriber_added_after_start_receives_events(bus):
    late = Recorder()
    bus.subscribe(Ping, late)

    bus.publish(Ping(n=1))
    await settle()

    assert late.events == [Ping(n=1)]


async def test_publish_threadsafe_delivers_from_another_thread(bus):
    pings = Recorder()
    bus.subscribe(Ping, pings)

    thread = threading.Thread(target=bus.publish_threadsafe, args=(Ping(n=7),))
    thread.start()
    thread.join()
    await settle()

    assert pings.events == [Ping(n=7)]


async def test_publish_from_another_thread_is_refused(bus):
    errors: list[BaseException] = []

    def call_publish() -> None:
        try:
            bus.publish(Ping(n=1))
        except RuntimeError as error:
            errors.append(error)

    thread = threading.Thread(target=call_publish)
    thread.start()
    thread.join()

    assert len(errors) == 1


async def test_publish_before_start_is_refused():
    bus = EventBus()

    with pytest.raises(RuntimeError):
        bus.publish(Ping(n=1))


async def test_close_handles_everything_queued_before_it():
    bus = EventBus()
    pings = Recorder()
    bus.subscribe(Ping, pings)
    await bus.start()

    for n in range(10):
        bus.publish(Ping(n=n))
    await bus.close()

    assert [e.n for e in pings.events] == list(range(10))


async def test_publish_after_close_is_ignored():
    bus = EventBus()
    pings = Recorder()
    bus.subscribe(Ping, pings)
    await bus.start()
    await bus.close()

    bus.publish(Ping(n=1))
    await settle()

    assert pings.events == []


async def test_full_queue_drops_the_oldest_event():
    bus = EventBus(queue_size=2)
    release = asyncio.Event()
    handled: list[int] = []

    async def blocked(event: Ping) -> None:
        await release.wait()
        handled.append(event.n)

    bus.subscribe(Ping, blocked)
    await bus.start()

    bus.publish(Ping(n=1))
    await settle()  # the worker takes 1 and blocks on it, the queue is empty again
    for n in (2, 3, 4):
        bus.publish(Ping(n=n))
    release.set()
    await bus.close()

    assert handled == [1, 3, 4]
