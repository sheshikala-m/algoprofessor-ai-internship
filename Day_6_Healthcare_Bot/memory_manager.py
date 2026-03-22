# ============================================================
# MEMORY MANAGER
# Day 6 Healthcare Bot: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Conversational memory for multi-turn data analytics
# ============================================================

# Without memory each LLM call is independent and the model
# cannot refer to what was said earlier in the conversation.
# Adding memory means storing previous messages and sending
# them back with every new request so the model maintains
# context across turns. This file implements three memory
# strategies used in production LLM applications:
#   BufferMemory   - keeps full history up to a token limit
#   WindowMemory   - keeps only the last N turns
#   SummaryMemory  - summarizes old turns to save tokens
# Each strategy has different tradeoffs between context length,
# token cost, and how much history the model can access.
# The ConversationManager class wraps any memory strategy and
# provides a clean chat interface that injects dataset context
# into every request automatically.

from datetime import datetime
from data_loader import load_titanic, summarize_dataframe
from openai_client import OpenAIClient


# ============================================================
# BASE MEMORY CLASS
# ============================================================

class BaseMemory:
    """
    Abstract base class for all memory strategies.
    All memory classes store messages as dicts with role and
    content keys matching the OpenAI and Anthropic API format.
    Subclasses override get_messages() to apply their strategy.
    """

    def __init__(self):
        self.messages = []
        self.strategy = "base"

    def add_user_message(self, content):
        """Adds a user turn to the message history."""
        self.messages.append({
            "role"     : "user",
            "content"  : content,
            "timestamp": datetime.now().isoformat()
        })

    def add_assistant_message(self, content):
        """Adds an assistant turn to the message history."""
        self.messages.append({
            "role"     : "assistant",
            "content"  : content,
            "timestamp": datetime.now().isoformat()
        })

    def get_messages(self):
        """Returns messages formatted for the LLM API."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
        ]

    def clear(self):
        """Clears all stored messages from memory."""
        self.messages = []
        print("Memory cleared.")

    def get_turn_count(self):
        """Returns the number of user turns stored."""
        return sum(1 for m in self.messages if m["role"] == "user")

    def get_token_estimate(self):
        """Estimates total tokens using 4 characters per token rule."""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4

    def summary(self):
        """Returns a one-line description of current memory state."""
        return (self.strategy + ": " + str(self.get_turn_count()) +
                " turns, ~" + str(self.get_token_estimate()) + " tokens")


# ============================================================
# BUFFER MEMORY
# ============================================================

class BufferMemory(BaseMemory):
    """
    Keeps the complete conversation history in memory up to a
    maximum token limit. When the limit is exceeded the oldest
    messages are removed in user-assistant pairs to keep the
    conversation structure intact. This is the simplest strategy
    and works well for short to medium length conversations.

    Parameters:
        max_tokens : int, approximate token limit for history
    """

    def __init__(self, max_tokens=2000):
        super().__init__()
        self.max_tokens = max_tokens
        self.strategy   = "BufferMemory"

    def get_messages(self):
        """Returns messages, removing oldest pairs if over token limit."""
        while (self.get_token_estimate() > self.max_tokens
               and len(self.messages) > 2):
            self.messages.pop(0)
            if self.messages and self.messages[0]["role"] == "assistant":
                self.messages.pop(0)

        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
        ]

    def summary(self):
        return (self.strategy + ": " + str(self.get_turn_count()) +
                " turns, ~" + str(self.get_token_estimate()) +
                " tokens, max=" + str(self.max_tokens))


# ============================================================
# WINDOW MEMORY
# ============================================================

class WindowMemory(BaseMemory):
    """
    Keeps only the last N conversation turns in memory.
    Older turns are discarded automatically. This is predictable
    in token usage and suitable when only recent context matters.

    Parameters:
        window_size : int, number of recent turns to keep
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.strategy    = "WindowMemory"

    def get_messages(self):
        """Returns only the most recent window_size turns."""
        recent = self.messages[-(self.window_size * 2):]
        return [
            {"role": m["role"], "content": m["content"]}
            for m in recent
        ]

    def summary(self):
        return (self.strategy + ": showing last " +
                str(self.window_size) + " turns of " +
                str(self.get_turn_count()) + " total turns stored")


# ============================================================
# SUMMARY MEMORY
# ============================================================

class SummaryMemory(BaseMemory):
    """
    Keeps a running LLM-generated summary of older turns plus
    the last max_full_turns turns in full text. When the history
    grows beyond max_full_turns the oldest turns are summarized
    and stored as a context block instead of verbatim text.
    This allows very long conversations without hitting token
    limits because old content is compressed not discarded.

    Parameters:
        max_full_turns : int, number of most recent turns kept in full
        client         : OpenAIClient instance used for summarization
    """

    def __init__(self, max_full_turns=4, client=None):
        super().__init__()
        self.max_full_turns  = max_full_turns
        self.client          = client
        self.running_summary = ""
        self.strategy        = "SummaryMemory"

    def _summarize_old_turns(self, old_messages):
        """
        Uses the LLM to compress old conversation turns into a
        2-3 sentence summary. Falls back to keyword extraction
        if no client is available.
        """
        if not old_messages:
            return ""

        conversation_text = "\n".join([
            m["role"].upper() + ": " + m["content"][:200]
            for m in old_messages
        ])

        if self.client:
            prompt = (
                "Summarize this data analysis conversation in 2-3 sentences, "
                "preserving all key statistics and findings mentioned:\n\n"
                + conversation_text
            )
            return self.client.complete(prompt, max_tokens=150)

        topics = set()
        for m in old_messages:
            for word in ["survival", "age", "class", "fare", "gender", "female", "male"]:
                if word in m["content"].lower():
                    topics.add(word)
        return (
            "Previous conversation covered: " + ", ".join(topics) + ". "
            "Key facts: Titanic dataset with 891 records, 38.4 percent survival rate."
        )

    def compress_if_needed(self):
        """
        Checks if stored history exceeds max_full_turns and
        compresses the oldest turns into the running summary.
        """
        user_turns = [m for m in self.messages if m["role"] == "user"]
        if len(user_turns) > self.max_full_turns:
            cutoff = len(self.messages) - (self.max_full_turns * 2)
            if cutoff > 0:
                old_messages         = self.messages[:cutoff]
                new_summary          = self._summarize_old_turns(old_messages)
                self.running_summary = new_summary
                self.messages        = self.messages[cutoff:]

    def get_messages(self):
        """Returns compressed summary context plus recent full turns."""
        self.compress_if_needed()
        result = []
        if self.running_summary:
            result.append({
                "role"   : "user",
                "content": "Earlier conversation summary: " + self.running_summary
            })
            result.append({
                "role"   : "assistant",
                "content": "Understood. I have the context from our earlier discussion."
            })
        result += [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
        ]
        return result

    def summary(self):
        return (self.strategy + ": " + str(self.get_turn_count()) +
                " turns, max_full=" + str(self.max_full_turns) +
                ", has_summary=" + str(bool(self.running_summary)))


# ============================================================
# CONVERSATION MANAGER
# ============================================================

class ConversationManager:
    """
    Manages a multi-turn analytics conversation using a chosen
    memory strategy. Injects the dataset summary as a system
    prompt so the model always has access to the data statistics
    without them being repeated in every user message.

    Parameters:
        memory_strategy : BaseMemory subclass instance
        client          : OpenAIClient or ClaudeClient instance
        dataset_summary : string, included in every system prompt
    """

    def __init__(self, memory_strategy, client, dataset_summary):
        self.memory  = memory_strategy
        self.client  = client
        self.summary = dataset_summary

        self.system_prompt = (
            "You are a data analyst assistant specializing in the Titanic "
            "passenger dataset. Use the following dataset statistics to answer "
            "questions accurately. Always cite specific numbers when making claims.\n\n"
            "Dataset Statistics:\n" + dataset_summary
        )
        print("ConversationManager ready with " +
              self.memory.strategy + " memory.")

    def chat(self, user_message):
        """
        Processes a user message, sends it to the LLM with full
        memory context, stores both turns, and returns the response.

        Parameters:
            user_message : string, the user question or statement

        Returns:
            string, the assistant response
        """
        self.memory.add_user_message(user_message)
        messages = self.memory.get_messages()

        system_msg    = [{"role": "system", "content": self.system_prompt}]
        full_messages = system_msg + messages
        response      = self.client.chat(full_messages)

        self.memory.add_assistant_message(response)
        return response

    def show_history(self):
        """Prints the full conversation history with memory summary."""
        print("\n-- Conversation History --")
        print(self.memory.summary())
        for i, msg in enumerate(self.memory.messages):
            role    = msg["role"].upper()
            content = msg["content"][:120]
            print("\n[" + str(i + 1) + "] " + role + ": " + content)

    def export_history(self):
        """Returns the full message history as a list of dicts."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.memory.messages
        ]


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("MEMORY MANAGER DEMO")
    print("=" * 55)

    client  = OpenAIClient()
    df      = load_titanic()
    summary = summarize_dataframe(df)

    questions = [
        "What is the overall survival rate in this dataset?",
        "Which gender had the higher survival rate?",
        "How about survival by passenger class?",
        "Based on what you told me, which group had the best odds?",
        "Can you summarize all the survival statistics we discussed?"
    ]

    print("\n-- Test 1: BufferMemory --")
    buffer_mem = BufferMemory(max_tokens=2000)
    manager1   = ConversationManager(buffer_mem, client, summary)
    for q in questions:
        print("\nUser     : " + q)
        response = manager1.chat(q)
        print("Assistant: " + response[:200])
    manager1.show_history()

    print("\n-- Test 2: WindowMemory (last 3 turns) --")
    window_mem = WindowMemory(window_size=3)
    manager2   = ConversationManager(window_mem, client, summary)
    for q in questions:
        print("\nUser     : " + q)
        response = manager2.chat(q)
        print("Assistant: " + response[:200])
    manager2.show_history()

    print("\n-- Test 3: SummaryMemory (max 2 full turns) --")
    summary_mem = SummaryMemory(max_full_turns=2, client=client)
    manager3    = ConversationManager(summary_mem, client, summary)
    for q in questions:
        print("\nUser     : " + q)
        response = manager3.chat(q)
        print("Assistant: " + response[:200])
    manager3.show_history()

    print("\n-- Memory Strategy Comparison --")
    print(buffer_mem.summary())
    print(window_mem.summary())
    print(summary_mem.summary())

    print("\n-- Memory Manager demo complete --")


if __name__ == "__main__":
    run_demo()
