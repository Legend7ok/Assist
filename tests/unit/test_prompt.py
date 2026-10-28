from assist.assistant.prompt import build_answer_prompt
from assist.core.schemas import Answer, Segment, Speaker

ITEMS = (
    Segment(speaker=Speaker.THEM, text="What is an index?", start=61, end=64),
    Answer(text="A structure that speeds up lookups.", at=65),
    Segment(speaker=Speaker.ME, text="It speeds up lookups.", start=66, end=69),
    Segment(speaker=Speaker.THEM, text="And its cost?", start=70, end=72),
)


def build(**overrides):
    parts = {"notes": "", "summary": "", "items": ITEMS, "request": ""} | overrides
    system, user = build_answer_prompt(**parts)
    return system.content, user.content


def test_parts_go_in_the_fixed_order():
    _, user = build(notes="Interview for Python role", summary="Talked about SQL", request="x")

    headers = [line for line in user.splitlines() if line.startswith("## ")]
    assert headers == [
        "## Meeting notes",
        "## Summary of the earlier part of the meeting",
        "## Recent conversation",
        "## One-off request from Me (highest priority)",
        "## Task",
    ]


def test_empty_parts_are_left_out():
    _, user = build(notes="  ", summary="", items=(), request="\n")

    assert user == "## Task\nAnswer the last question of the Interlocutor."


def test_rules_ask_for_a_short_answer_without_a_request():
    system, user = build()

    assert "Be short" in system
    assert "Answer the last question of the Interlocutor." in user


def test_one_off_request_lifts_the_length_limit():
    system, user = build(request="Review this code:\ndef f(): pass")

    assert "Be short" not in system
    assert "the length limit does not apply" in system
    assert "Review this code:\ndef f(): pass" in user
    assert user.endswith("## Task\nDo the one-off request above.")


def test_earlier_answers_stand_in_time_among_the_phrases():
    _, user = build()

    assert (
        "## Recent conversation\n"
        "[01:01] Interlocutor: What is an index?\n"
        "[01:05] Prompter answer: A structure that speeds up lookups.\n"
        "[01:06] Me: It speeds up lookups.\n"
        "[01:10] Interlocutor: And its cost?"
    ) in user
