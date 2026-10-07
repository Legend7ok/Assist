import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import structlog

from assist.core.events import Event

log = structlog.get_logger(__name__)

type Handler[E: Event] = Callable[[E], Awaitable[None]]

# Put into a subscriber's queue by close(): everything queued before it is still handled.
_STOP = object()


@dataclass(eq=False)
class _Subscription:
    event_type: type[Event]
    handler: Handler
    name: str
    queue: asyncio.Queue[object]
    task: asyncio.Task[None] | None = field(default=None)


class EventBus:
    """In-process publish/subscribe between the parts of the app.

    Every subscriber has its own queue and worker task, so a slow subscriber delays only
    itself, and events reach each subscriber in the order they were published. A subscriber
    to a base class (for example Event) receives all of its subclasses.
    """

    def __init__(self, *, queue_size: int = 1000) -> None:
        self._queue_size = queue_size
        self._subscriptions: list[_Subscription] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def subscribe[E: Event](
        self, event_type: type[E], handler: Handler[E], *, name: str | None = None
    ) -> None:
        # After close() nobody sends this worker its stop marker: it would wait forever.
        if self._closed:
            raise RuntimeError("Event bus is closed")
        subscription = _Subscription(
            event_type=event_type,
            handler=handler,
            name=name or getattr(handler, "__qualname__", repr(handler)),
            queue=asyncio.Queue(maxsize=self._queue_size),
        )
        self._subscriptions.append(subscription)
        if self._loop is not None:
            self._start_worker(subscription)

    async def start(self) -> None:
        if self._loop is not None:
            raise RuntimeError("Event bus is already started")
        self._loop = asyncio.get_running_loop()
        for subscription in self._subscriptions:
            self._start_worker(subscription)

    def publish(self, event: Event) -> None:
        """Publish from the bus's event loop thread. Other threads use publish_threadsafe."""
        if self._loop is None:
            raise RuntimeError("Event bus is not started")
        # asyncio.Queue is not thread-safe: a call from a foreign thread would corrupt it
        # silently instead of failing, so it is refused here.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not self._loop:
            raise RuntimeError("publish() called outside the bus loop, use publish_threadsafe()")

        if self._closed:
            log.warning("bus_publish_after_close", program_event=type(event).__name__)
            return

        for subscription in self._subscriptions:
            if isinstance(event, subscription.event_type):
                self._enqueue(subscription, event)

    def publish_threadsafe(self, event: Event) -> None:
        """Publish from any thread: audio callbacks, the pywebview window, the hotkey listener."""
        if self._loop is None:
            raise RuntimeError("Event bus is not started")
        try:
            self._loop.call_soon_threadsafe(self.publish, event)
        except RuntimeError:
            # The loop is already closed: an audio callback can still fire during shutdown.
            log.debug("bus_publish_after_loop_closed", program_event=type(event).__name__)

    async def close(self, grace_period: float = 5.0) -> None:
        """Stop accepting events and let subscribers finish their queues.

        A subscriber still busy after `grace_period` seconds (a handler stuck on a dead
        network connection) is cancelled, so closing the app never hangs on it.
        """
        if self._closed:
            return
        self._closed = True

        stop_markers: list[asyncio.Task[None]] = []
        for subscription in self._subscriptions:
            if subscription.queue.full():
                # Waiting here for room could block forever on a stalled subscriber. The put
                # waits in the background instead and is bounded by the grace period below.
                stop_markers.append(asyncio.create_task(subscription.queue.put(_STOP)))
            else:
                subscription.queue.put_nowait(_STOP)

        workers = {s.task: s for s in self._subscriptions if s.task is not None}
        pending: set[asyncio.Task[None]] = set()
        if workers:
            _, pending = await asyncio.wait(workers, timeout=grace_period)
        for task in pending:
            log.warning("bus_subscriber_cancelled_on_close", subscriber=workers[task].name)
            task.cancel()
        for marker in stop_markers:
            marker.cancel()
        await asyncio.gather(*pending, *stop_markers, return_exceptions=True)

    def _start_worker(self, subscription: _Subscription) -> None:
        assert self._loop is not None
        subscription.task = self._loop.create_task(
            self._run(subscription), name=f"bus:{subscription.name}"
        )

    def _enqueue(self, subscription: _Subscription, event: Event) -> None:
        if subscription.queue.full():
            # A stalled subscriber must not grow memory without bound or block publishers
            # (audio callbacks publish ~50 frames per second). The oldest event is the least
            # useful one to keep.
            dropped = subscription.queue.get_nowait()
            log.warning(
                "bus_queue_overflow",
                subscriber=subscription.name,
                dropped=type(dropped).__name__,
            )
        subscription.queue.put_nowait(event)

    async def _run(self, subscription: _Subscription) -> None:
        while True:
            item = await subscription.queue.get()
            if item is _STOP:
                return
            try:
                await subscription.handler(item)
            except Exception:
                log.exception(
                    "bus_handler_failed",
                    subscriber=subscription.name,
                    program_event=type(item).__name__,
                )
