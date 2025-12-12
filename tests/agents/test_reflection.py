"""Comprehensive tests for reflection components."""

from datetime import datetime, timezone

from corral.agents.reflection import (
    Reflection,
    ReflectionMemory,
    ReflectionModule,
    create_reflexion_history,
)
from corral.agents.utils import LiteLLMMessage


class TestReflection:
    """Test cases for the Reflection dataclass."""

    def test_reflection_creation(self):
        """Test creating a Reflection object."""
        trajectory = [
            LiteLLMMessage(role="user", content="Test task"),
            LiteLLMMessage(role="assistant", content="Test response"),
        ]

        reflection = Reflection(
            trial_index=0,
            task_id="test_task",
            trajectory=trajectory,
            reflection_text="I should check tool parameters before calling.",
            score=0.3,
        )

        assert reflection.trial_index == 0
        assert reflection.task_id == "test_task"
        assert len(reflection.trajectory) == 2
        assert (
            reflection.reflection_text
            == "I should check tool parameters before calling."
        )
        assert reflection.score == 0.3
        assert isinstance(reflection.timestamp, datetime)

    def test_reflection_to_dict(self):
        """Test converting reflection to dictionary."""
        trajectory = [
            LiteLLMMessage(role="user", content="Test task"),
        ]

        reflection = Reflection(
            trial_index=1,
            task_id="task_1",
            trajectory=trajectory,
            reflection_text="Reflection text",
            score=0.5,
        )

        data = reflection.to_dict()

        assert data["trial_index"] == 1
        assert data["task_id"] == "task_1"
        assert data["reflection_text"] == "Reflection text"
        assert data["score"] == 0.5
        assert "timestamp" in data
        assert len(data["trajectory"]) == 1

    def test_reflection_from_dict(self):
        """Test creating reflection from dictionary."""
        data = {
            "trial_index": 2,
            "task_id": "task_2",
            "trajectory": [
                {"role": "user", "content": "Test"},
            ],
            "reflection_text": "Test reflection",
            "score": 0.7,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        reflection = Reflection.from_dict(data)

        assert reflection.trial_index == 2
        assert reflection.task_id == "task_2"
        assert len(reflection.trajectory) == 1
        assert reflection.reflection_text == "Test reflection"
        assert reflection.score == 0.7


class TestReflectionMemory:
    """Test cases for ReflectionMemory."""

    def test_memory_initialization(self):
        """Test initializing memory with default size."""
        memory = ReflectionMemory()

        assert memory.max_size == 3
        assert len(memory.reflections) == 0

    def test_memory_custom_size(self):
        """Test initializing memory with custom size."""
        memory = ReflectionMemory(max_size=5)

        assert memory.max_size == 5

    def test_add_reflection(self):
        """Test adding reflections to memory."""
        memory = ReflectionMemory(max_size=3)

        reflection1 = self._create_test_reflection(0, "task_1", "First reflection")
        memory.add_reflection(reflection1)

        assert len(memory.reflections) == 1
        assert memory.reflections[0].reflection_text == "First reflection"

    def test_fifo_behavior(self):
        """Test that memory maintains FIFO when exceeding max_size."""
        memory = ReflectionMemory(max_size=3)

        # Add 4 reflections
        for i in range(4):
            reflection = self._create_test_reflection(i, "task_1", f"Reflection {i}")
            memory.add_reflection(reflection)

        # Should only keep last 3
        assert len(memory.reflections) == 3
        assert memory.reflections[0].reflection_text == "Reflection 1"
        assert memory.reflections[1].reflection_text == "Reflection 2"
        assert memory.reflections[2].reflection_text == "Reflection 3"

    def test_clear_memory(self):
        """Test clearing all reflections."""
        memory = ReflectionMemory()

        # Add some reflections
        for i in range(2):
            reflection = self._create_test_reflection(i, "task_1", f"Reflection {i}")
            memory.add_reflection(reflection)

        assert len(memory.reflections) == 2

        memory.clear()

        assert len(memory.reflections) == 0

    def test_format_for_prompt_empty(self):
        """Test formatting empty memory."""
        memory = ReflectionMemory()

        formatted = memory.format_for_prompt()

        assert formatted == ""

    def test_format_for_prompt_single(self):
        """Test formatting single reflection."""
        memory = ReflectionMemory()
        reflection = self._create_test_reflection(
            0, "task_1", "Check parameters before calling tools."
        )
        memory.add_reflection(reflection)

        formatted = memory.format_for_prompt()

        assert "Attempt 1" in formatted
        assert "Score: 0.30" in formatted
        assert "Check parameters before calling tools." in formatted

    def test_format_for_prompt_multiple(self):
        """Test formatting multiple reflections."""
        memory = ReflectionMemory()

        reflection1 = self._create_test_reflection(0, "task_1", "First lesson")
        reflection2 = self._create_test_reflection(1, "task_1", "Second lesson")

        memory.add_reflection(reflection1)
        memory.add_reflection(reflection2)

        formatted = memory.format_for_prompt()

        assert "Attempt 1" in formatted
        assert "First lesson" in formatted
        assert "Attempt 2" in formatted
        assert "Second lesson" in formatted

    def test_memory_serialization(self):
        """Test converting memory to/from dict."""
        memory = ReflectionMemory(max_size=2)

        reflection = self._create_test_reflection(0, "task_1", "Test reflection")
        memory.add_reflection(reflection)

        # Convert to dict
        data = memory.to_dict()

        assert data["max_size"] == 2
        assert len(data["reflections"]) == 1

        # Convert back
        restored_memory = ReflectionMemory.from_dict(data)

        assert restored_memory.max_size == 2
        assert len(restored_memory.reflections) == 1
        assert restored_memory.reflections[0].reflection_text == "Test reflection"

    @staticmethod
    def _create_test_reflection(
        trial_index: int, task_id: str, reflection_text: str
    ) -> Reflection:
        """Helper to create a test reflection."""
        return Reflection(
            trial_index=trial_index,
            task_id=task_id,
            trajectory=[LiteLLMMessage(role="user", content="Test")],
            reflection_text=reflection_text,
            score=0.3,
        )


class TestReflectionModule:
    """Test cases for ReflectionModule."""

    def test_module_initialization(self):
        """Test initializing reflection module."""
        module = ReflectionModule(
            model="test-model", reflection_prompt="Generate reflection: {{trajectory}}"
        )

        assert module.model == "test-model"
        assert module.temperature == 0.0  # Deterministic by default

    def test_module_custom_temperature(self):
        """Test module with custom temperature."""
        module = ReflectionModule(
            model="test-model",
            reflection_prompt="Generate reflection: {{trajectory}}",
            temperature=0.5,
        )

        assert module.temperature == 0.5

    def test_default_prompt_exists(self):
        """Test that default prompt is available."""
        module = ReflectionModule(
            model="test-model", reflection_prompt="Generate reflection: {{trajectory}}"
        )

        assert module.reflection_prompt is not None
        assert len(module.reflection_prompt) > 0

    def test_summarize_trajectory_short(self):
        """Test trajectory summarization with few messages."""
        module = ReflectionModule(
            model="test-model", reflection_prompt="Generate reflection: {{trajectory}}"
        )

        trajectory = [
            LiteLLMMessage(role="user", content="Task"),
            LiteLLMMessage(role="assistant", content="Response"),
        ]

        summary = module._summarize_trajectory(trajectory)

        assert "USER: Task" in summary
        assert "ASSISTANT: Response" in summary

    def test_summarize_trajectory_long(self):
        """Test trajectory summarization with many messages."""
        module = ReflectionModule(
            model="test-model", reflection_prompt="Generate reflection: {{trajectory}}"
        )

        # Create 20 messages
        trajectory = [
            LiteLLMMessage(
                role="user" if i % 2 == 0 else "assistant", content=f"Message {i}"
            )
            for i in range(20)
        ]

        summary = module._summarize_trajectory(trajectory)

        # Should include all messages as summarization is for tool output, not message count
        assert "Message 0" in summary
        assert "Message 4" in summary
        assert "Message 15" in summary
        assert "Message 19" in summary

    def test_format_messages(self):
        """Test formatting messages."""
        module = ReflectionModule(
            model="test-model", reflection_prompt="Generate reflection: {{trajectory}}"
        )

        messages = [
            LiteLLMMessage(role="user", content="Hello"),
            LiteLLMMessage(role="assistant", content="Hi there"),
        ]

        formatted = module._format_messages(messages)

        assert "USER: Hello" in formatted
        assert "ASSISTANT: Hi there" in formatted

    def test_format_messages_truncation(self):
        """Test that tool messages are summarized."""
        module = ReflectionModule(
            model="test-model", reflection_prompt="Generate reflection: {{trajectory}}"
        )

        # Use a tool-like message (role="tool")
        long_content = "x" * 600  # Long tool output
        messages = [
            LiteLLMMessage(role="tool", content=long_content, tool_call_id="123"),
        ]

        formatted = module._format_messages(messages)

        # Tool messages should be summarized
        assert "TOOL:" in formatted


class TestCreateReflexionHistory:
    """Test cases for create_reflexion_history function."""

    def test_empty_memory(self):
        """Test with empty memory."""
        memory = ReflectionMemory()

        history = create_reflexion_history(memory)

        assert history is None

    def test_with_reflections(self):
        """Test with reflections in memory."""
        memory = ReflectionMemory()

        reflection = Reflection(
            trial_index=0,
            task_id="task_1",
            trajectory=[LiteLLMMessage(role="user", content="Test")],
            reflection_text="Learn from this mistake",
            score=0.2,
        )
        memory.add_reflection(reflection)

        history = create_reflexion_history(memory)

        assert history is not None
        assert len(history) == 1
        assert history[0]["role"] == "system"
        assert "LESSONS FROM PREVIOUS ATTEMPTS" in history[0]["content"]
        assert "Learn from this mistake" in history[0]["content"]
        assert history[0].get("name") == "reflexion-memory"

    def test_multiple_reflections(self):
        """Test with multiple reflections."""
        memory = ReflectionMemory()

        for i in range(2):
            reflection = Reflection(
                trial_index=i,
                task_id="task_1",
                trajectory=[LiteLLMMessage(role="user", content="Test")],
                reflection_text=f"Lesson {i + 1}",
                score=0.2,
            )
            memory.add_reflection(reflection)

        history = create_reflexion_history(memory)

        assert history is not None
        assert "Lesson 1" in history[0]["content"]
        assert "Lesson 2" in history[0]["content"]
