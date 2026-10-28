import structlog

from assist.assistant.prompts import load_prompt
from assist.context.window import ContextItem, format_timeline
from assist.core.schemas import Message

log = structlog.get_logger(__name__)

_SHORT = "Be short: 1-3 bullet points with only the key points."
_NO_LIMIT = (
    "Me has a one-off request: do it in full, the length limit does not apply "
    "(for example, a complete code review with code blocks)."
)
_ANSWER_TASK = "Answer the last question of the Interlocutor."
_REQUEST_TASK = "Do the one-off request above."

# Rough: in the GPT tokenizers a token is about 4 characters of English and 2-3 of Ukrainian.
# It is only for the log, to see how close a prompt comes to the provider's per-minute limit.
_CHARS_PER_TOKEN = 3


def build_answer_prompt(
    *, notes: str, summary: str, items: tuple[ContextItem, ...], request: str
) -> list[Message]:
    """Messages for one answer, top to bottom in the order the working plan fixes.

    Rules, meeting notes, summary, recent conversation with earlier answers in their place in
    time, the one-off request, the task. Empty parts are left out.
    """
    notes, summary, request = notes.strip(), summary.strip(), request.strip()
    rules = load_prompt("answer").format(length_rule=_NO_LIMIT if request else _SHORT)

    sections = [
        ("Meeting notes", notes),
        ("Summary of the earlier part of the meeting", summary),
        ("Recent conversation", format_timeline(items)),
        ("One-off request from Me (highest priority)", request),
        ("Task", _REQUEST_TASK if request else _ANSWER_TASK),
    ]
    context = "\n\n".join(f"## {title}\n{body}" for title, body in sections if body)

    messages = [Message(role="system", content=rules), Message(role="user", content=context)]
    log.info(
        "answer_prompt_built",
        estimated_tokens=sum(len(m.content) for m in messages) // _CHARS_PER_TOKEN,
        window_items=len(items),
        has_summary=bool(summary),
        has_notes=bool(notes),
        has_request=bool(request),
    )
    return messages
