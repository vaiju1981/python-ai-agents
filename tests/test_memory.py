import anyio

from python_ai_agents import (
    AgentRequest,
    DefaultAgent,
    InMemoryConversationStore,
    Message,
    ModelRequest,
    ModelResponse,
    RequestContext,
    SQLiteConversationStore,
    WindowedMemory,
)


class MemoryEchoModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def chat(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        users = [message.content for message in request.messages if message.role.value == "user"]
        return ModelResponse.text_response(" / ".join(users))


def test_default_agent_remembers_conversation_by_session() -> None:
    async def run() -> None:
        model = MemoryEchoModel()
        store = InMemoryConversationStore()
        agent = DefaultAgent(model, conversation_store=store)
        context = RequestContext(session_id="session-1", tenant="tenant-a")

        first = await agent.run(AgentRequest("first", context))
        second = await agent.run(AgentRequest("second", context))

        assert first.output == "first"
        assert second.output == "first / second"
        assert store.messages("tenant-a", "session-1") == (
            Message.user("first"),
            Message.assistant("first"),
            Message.user("second"),
            Message.assistant("first / second"),
        )

    anyio.run(run)


def test_default_agent_isolates_conversation_by_session() -> None:
    async def run() -> None:
        model = MemoryEchoModel()
        agent = DefaultAgent(model, conversation_store=InMemoryConversationStore())

        await agent.run(AgentRequest("one", RequestContext(session_id="a", tenant="tenant-a")))
        response = await agent.run(
            AgentRequest("two", RequestContext(session_id="b", tenant="tenant-a"))
        )

        assert response.output == "two"

    anyio.run(run)


def test_default_agent_can_run_without_remembering_conversation() -> None:
    async def run() -> None:
        model = MemoryEchoModel()
        agent = DefaultAgent(model, remember_conversation=False)
        context = RequestContext(session_id="session-1", tenant="tenant-a")

        await agent.run(AgentRequest("first", context))
        response = await agent.run(AgentRequest("second", context))

        assert response.output == "second"

    anyio.run(run)


def test_windowed_memory_keeps_system_and_recent_messages() -> None:
    memory = WindowedMemory(max_recent=2)

    memory.add(Message.system("system"))
    memory.add(Message.user("one"))
    memory.add(Message.assistant("two"))
    memory.add(Message.user("three"))

    assert memory.history() == (
        Message.system("system"),
        Message.assistant("two"),
        Message.user("three"),
    )


def test_in_memory_conversation_store_lists_deletes_and_evicts() -> None:
    async def run() -> None:
        store = InMemoryConversationStore(max_sessions=1)

        async with store.memory("tenant-a", "session-1") as memory:
            memory.add(Message.user("one"))
        async with store.memory("tenant-a", "session-2") as memory:
            memory.add(Message.user("two"))

        assert store.messages("tenant-a", "session-1") == ()
        assert store.messages("tenant-a", "session-2") == (Message.user("two"),)
        assert [session.session_id for session in store.list_sessions("tenant-a")] == ["session-2"]

        store.delete("tenant-a", "session-2")
        assert store.list_sessions("tenant-a") == []

    anyio.run(run)


def test_sqlite_conversation_store_persists_across_instances(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "conversation.sqlite3"
        store = SQLiteConversationStore(path)
        agent = DefaultAgent(MemoryEchoModel(), conversation_store=store)
        context = RequestContext(session_id="session-1", tenant="tenant-a")

        await agent.run(AgentRequest("first", context))
        await agent.run(AgentRequest("second", context))

        restored = SQLiteConversationStore(path)
        assert restored.messages("tenant-a", "session-1") == (
            Message.user("first"),
            Message.assistant("first"),
            Message.user("second"),
            Message.assistant("first / second"),
        )
        assert [session.session_id for session in restored.list_sessions("tenant-a")] == [
            "session-1"
        ]

    anyio.run(run)


def test_sqlite_conversation_store_deletes_session(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteConversationStore(tmp_path / "conversation.sqlite3")

        async with store.memory("tenant-a", "session-1") as memory:
            memory.add(Message.user("hello"))
        store.delete("tenant-a", "session-1")

        assert store.messages("tenant-a", "session-1") == ()
        assert store.list_sessions("tenant-a") == []

    anyio.run(run)
