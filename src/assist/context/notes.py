from assist.core.bus import EventBus
from assist.core.events import NotesChanged, RequestChanged


class MeetingNotes:
    """The two extra fields as the user last typed them: meeting notes and one-off request."""

    def __init__(self, bus: EventBus) -> None:
        self._notes = ""
        self._request = ""
        bus.subscribe(NotesChanged, self._on_notes, name="meeting_notes")
        bus.subscribe(RequestChanged, self._on_request, name="meeting_request")

    @property
    def notes(self) -> str:
        return self._notes

    @property
    def request(self) -> str:
        return self._request

    async def _on_notes(self, event: NotesChanged) -> None:
        self._notes = event.text

    async def _on_request(self, event: RequestChanged) -> None:
        self._request = event.text
