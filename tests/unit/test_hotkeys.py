import asyncio
import threading

from pynput.keyboard import Key, KeyCode

from assist.core.bus import EventBus
from assist.core.events import AnswerRequested
from assist.desktop.hotkeys import AltGrTap, AnswerHotkey

Q = KeyCode.from_char("q")


def taps(*events: tuple[str, object]) -> list[bool]:
    """Feeds ("down" / "up", key) events and returns what each release reported."""
    tap = AltGrTap()
    results = []
    for kind, key in events:
        if kind == "down":
            tap.press(key)
        else:
            results.append(tap.release(key))
    return results


def test_lone_alt_gr_without_fake_ctrl_is_a_tap():
    assert taps(("down", Key.alt_gr), ("up", Key.alt_gr)) == [True]


def test_fake_left_ctrl_sent_with_alt_gr_is_not_another_key():
    assert taps(
        ("down", Key.ctrl_l),
        ("down", Key.alt_gr),
        ("up", Key.ctrl_l),
        ("up", Key.alt_gr),
    ) == [False, True]


def test_held_alt_gr_with_repeats_is_still_a_tap():
    repeats = [("down", Key.ctrl_l), ("down", Key.alt_gr)] * 3

    assert taps(*repeats, ("up", Key.ctrl_l), ("up", Key.alt_gr)) == [False, True]


def test_alt_gr_used_to_type_a_character_is_not_a_tap():
    assert taps(
        ("down", Key.ctrl_l),
        ("down", Key.alt_gr),
        ("down", Q),
        ("up", Q),
        ("up", Key.ctrl_l),
        ("up", Key.alt_gr),
    ) == [False, False, False]


def test_tap_after_a_character_combination_works():
    assert taps(
        ("down", Key.alt_gr),
        ("down", Q),
        ("up", Q),
        ("up", Key.alt_gr),
        ("down", Key.alt_gr),
        ("up", Key.alt_gr),
    ) == [False, False, True]


def test_other_keys_alone_are_never_a_tap():
    assert taps(
        ("down", Key.alt_l), ("up", Key.alt_l), ("down", Q), ("up", Q), ("up", Key.alt_gr)
    ) == [False, False, False]


class FakeListener:
    """Stands in for pynput's Listener and keeps its callbacks so a test can press keys."""

    def __init__(self, *, on_press, on_release) -> None:
        self.on_press = on_press
        self.on_release = on_release


async def test_tap_publishes_answer_requested_from_the_hook_thread():
    bus = EventBus()
    requests: list[AnswerRequested] = []

    async def record(event: AnswerRequested) -> None:
        requests.append(event)

    bus.subscribe(AnswerRequested, record)
    await bus.start()
    listeners: list[FakeListener] = []

    def factory(**callbacks) -> FakeListener:
        listeners.append(FakeListener(**callbacks))
        return listeners[-1]

    AnswerHotkey(bus, listener_factory=factory)
    (listener,) = listeners

    def press_alt_gr() -> None:
        listener.on_press(Key.ctrl_l, False)
        listener.on_press(Key.alt_gr, False)
        listener.on_release(Key.ctrl_l, False)
        listener.on_release(Key.alt_gr, False)

    thread = threading.Thread(target=press_alt_gr)
    thread.start()
    thread.join()
    for _ in range(5):
        await asyncio.sleep(0)
    await bus.close()

    assert requests == [AnswerRequested()]
