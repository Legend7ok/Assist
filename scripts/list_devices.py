"""List WASAPI audio devices and show which ones the config picks.

Usage: uv run python scripts/list_devices.py
Names printed here go to `loopback_device` / `mic_device` in config/default.toml.
"""

from collections.abc import Callable

import pyaudiowpatch as pyaudio

from assist.audio.capture import AudioDevice, AudioDeviceError, DeviceCatalog
from assist.settings import Settings


def describe(device: AudioDevice, catalog: DeviceCatalog) -> str:
    if device.is_loopback:
        kind = "loopback"
    elif device.input_channels > 0:
        kind = "microphone"
    else:
        kind = "output"
    channels = f"{device.input_channels} ch" if device.input_channels else "    "
    default = ""
    if device.index in (catalog.default_input_index, catalog.default_output_index):
        default = "  (Windows default)"
    return (
        f"  [{device.index:>3}] {kind:<10} {device.sample_rate:>6} Hz {channels}  "
        f"{device.name}{default}"
    )


def picked(choose: Callable[[str | None], AudioDevice], name: str | None) -> str:
    try:
        return repr(choose(name).name)
    except AudioDeviceError as error:
        return f"error: {error}"


def main() -> None:
    settings = Settings().audio
    pa = pyaudio.PyAudio()
    try:
        catalog = DeviceCatalog.from_pyaudio(pa)
    finally:
        pa.terminate()

    print("WASAPI audio devices:")
    for device in catalog.devices:
        print(describe(device, catalog))

    print("\nPicked by the config:")
    print(f"  them (loopback):  {picked(catalog.loopback, settings.loopback_device)}")
    print(f"  me (microphone):  {picked(catalog.microphone, settings.mic_device)}")


if __name__ == "__main__":
    main()
