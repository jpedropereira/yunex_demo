"""Chat tests. No running stack required -- only the message assembly is checked."""

from yunex_demo.chat import SYSTEM_PROMPT, build_messages


def test_system_prompt_is_the_plain_baseline():
    assert SYSTEM_PROMPT == "You are a helpful assistant"


def test_system_prompt_comes_first_and_history_is_kept_in_order():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    messages = build_messages(history)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:] == history


def test_build_messages_does_not_mutate_the_history():
    history = [{"role": "user", "content": "Hi"}]
    build_messages(history)
    assert history == [{"role": "user", "content": "Hi"}]
