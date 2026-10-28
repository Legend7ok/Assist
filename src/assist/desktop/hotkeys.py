from collections.abc import Callable
from typing import Any

import structlog
from pynput import keyboard

from assist.core.bus import EventBus
from assist.core.events import AnswerRequested

log = structlog.get_logger(__name__)


class AltGrTap:
    """Tells a lone AltGr press from AltGr used to type a character (AltGr+q gives @).

    Fed with pynput keys in the order they happen; `release` returns True when AltGr went up
    with no other key pressed while it was down.
    """

    def __init__(self) -> None:
        self._down = False
        self._alone = False

    def press(self, key: object) -> None:
        if key == keyboard.Key.alt_gr:
            # Held AltGr repeats its press; only the first one starts a tap.
            if not self._down:
                self._down = True
                self._alone = True
        # On layouts with AltGr (Ukrainian too) Windows sends a fake left Ctrl together with
        # every AltGr press and repeat. It cannot be told from a real left Ctrl, so left Ctrl
        # never counts as another key.
        elif key != keyboard.Key.ctrl_l:
            self._alone = False

    def release(self, key: object) -> bool:
        if key != keyboard.Key.alt_gr or not self._down:
            return False
        self._down = False
        return self._alone


type ListenerFactory = Callable[..., Any]


class AnswerHotkey:
    """Publishes AnswerRequested on a lone AltGr press anywhere in the system.

    pynput calls back from its own keyboard hook thread, so events go through
    publish_threadsafe. Start it after the bus has started.
    """

    def __init__(
        self, bus: EventBus, *, listener_factory: ListenerFactory = keyboard.Listener
    ) -> None:
        self._bus = bus
        self._tap = AltGrTap()
        self._listener = listener_factory(on_press=self._on_press, on_release=self._on_release)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def _on_press(self, key: object, injected: bool = False) -> None:
        self._tap.press(key)

    def _on_release(self, key: object, injected: bool = False) -> None:
        if self._tap.release(key):
            log.info("hotkey_pressed")
            self._bus.publish_threadsafe(AnswerRequested())
