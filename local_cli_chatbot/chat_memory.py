from collections import deque

class SlidingWindowMemory:
    def __init__(self, max_turns: int = 3):
        """
        max_turns: number of recent *user+bot* exchanges to keep.
        Example: max_turns=3 keeps last 3 user messages and last 3 bot replies.
        """
        self.max_turns = max_turns
        # store tuples (user_text, bot_text)
        self.buffer = deque(maxlen=max_turns)

    def add_turn(self, user_text: str, bot_text: str):
        """Add a completed exchange to memory."""
        self.buffer.append((user_text.strip(), bot_text.strip()))

    def get_prompt(self, current_user_input: str, system_prompt: str = None) -> str:
        """
        Build a prompt string that includes recent history + current user input.
        We format turns as:
        User: ...
        Bot: ...
        User: <current_user_input>
        Bot:
        """
        parts = []
        if system_prompt:
            parts.append(system_prompt.strip())
        for user_text, bot_text in self.buffer:
            parts.append(f"User: {user_text}")
            parts.append(f"Bot: {bot_text}")
        parts.append(f"User: {current_user_input.strip()}")
        parts.append("Bot:")
        # Join with newlines to make the prompt readable
        return "\n".join(parts)

    def clear(self):
        self.buffer.clear()
