from __future__ import annotations

"""
AgentCore-backed Session for the OpenAI Agents SDK
--------------------------------------------------

Implements `SessionABC` using the Amazon Bedrock AgentCore Starter Toolkit
(`bedrock_agentcore`) memory features

Short‑term memory is backed by AgentCore Events. Optional helpers are provided
for long‑term semantic memory retrieval.

Requirements
~~~~~~~~~~~~
    pip install openai openai-agents-python bedrock-agentcore-starter-toolkit

Usage (basic)
~~~~~~~~~~~~~
    from agents import Agent, Runner
    from agentcore_session import AgentCoreSession

    session = AgentCoreSession(
        memory_id="mem-abc123",        # Your AgentCore Memory resource ID
        session_id="thread_42",       # Your conversation/thread id
        actor_id="user-123",          # End-user identifier
        region="us-west-2",           # AgentCore region
    )

    result = Runner.run_sync(Agent("Assistant"), "Hello!", session=session)
    print(result.final_output)

Usage (with pop/corrections)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Undo the last message (assistant or user)
    removed = asyncio.run(session.pop_item())
    # Now the next add will branch from the previous event transparently

Optional: inject long‑term memories for a query
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build extra input items containing top‑k retrieved facts for the question
    extra_context_items = session.build_long_term_context(
        namespace="support/facts/{sessionId}",
        query="What did we decide about pagination?",
        top_k=3,
    )
    # Then pass to the runner along with the user input
    result = Runner.run_sync(
        Agent("Assistant"),
        [*extra_context_items, {"role": "user", "content": "What did we decide about pagination?"}],
        session=session,
    )

Notes
~~~~~
- `clear_session()` clears *this Session's* view of history by ignoring prior
  events; AgentCore events are immutable and are not deleted upstream.
- `pop_item()` is implemented as a *branch-on-next-add* optimization: we mark
  the previous event as the fork root and continue on a fresh branch on your
  next `add_items()` call. No upstream deletion is performed.
- Each message is stored as its own Event to allow granular pop semantics.

"""

import asyncio
import datetime as dt
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agents.items import TResponseInputItem
from agents.memory.session import SessionABC

try:
    # Starter Toolkit high-level client (wraps boto3 internally so you don't have to)
    from bedrock_agentcore.memory import MemoryClient
except Exception as e:  # pragma: no cover - import-time error is clearer to the caller
    raise ImportError(
        "bedrock-agentcore-starter-toolkit is required: `pip install bedrock-agentcore-starter-toolkit`"
    ) from e


ResponseItem = Dict[str, Any]


def _content_part_for_role(text: str, role: str) -> Dict[str, str]:
    """Return the correct Responses content part based on role.
    - user/developer/system → input_text
    - assistant/tool/other → output_text
    """
    r = (role or "").lower()
    if r in ("user", "system", "developer"):
        return {"type": "input_text", "text": text}
    return {"type": "output_text", "text": text}


class AgentCoreSession(SessionABC):
    """OpenAI Agents SDK Session backed by Amazon AgentCore Memory.

    Parameters
    ----------
    memory_id : str
        AgentCore Memory resource ID (control plane identifier).
    session_id : str
        Logical conversation/thread id. Used for grouping AgentCore Events.
    actor_id : str
        End-user identifier used by AgentCore.
    region : str | None
        AWS region for the Memory service. If omitted, MemoryClient picks
        the default from your environment.
    branch_name : str | None
        Optional explicit branch name to continue on. When `pop_item()` is used,
        a new branch will be created automatically (name prefixed with "fix-").
    client : MemoryClient | None
        Optionally pass an already-configured MemoryClient.
    """

    # ------------- Construction -------------
    def __init__(
        self,
        memory_id: str,
        session_id: str,
        actor_id: str,
        region: Optional[str] = None,
        branch_name: Optional[str] = None,
        client: Optional[MemoryClient] = None,
    ) -> None:
        self.memory_id = memory_id
        self.session_id = session_id
        self.actor_id = actor_id
        self._client = client or MemoryClient(region_name=region)

        # View/state controls for pop/clear/branching
        self._current_branch: Optional[str] = branch_name
        self._pop_fork_root_event_id: Optional[str] = None  # set when pop_item() called
        self._cleared: bool = False  # when True, get_items() returns [] until new adds

    # ------------- SessionABC methods -------------
    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        """Return conversation history as OpenAI Responses items.

        Items are produced in chronological order. If `clear_session()` was
        called, we return an empty list until new items are added.
        """
        if self._cleared:
            return []

        events = await asyncio.to_thread(
            self._client.list_events,
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            branch_name=self._current_branch or None,
            include_parent_events=False,
            max_results=limit or 100,
            include_payload=True,
        )

        # If a pop was requested, hide events after the fork root
        if self._pop_fork_root_event_id:
            try:
                idx = next(i for i, e in enumerate(events) if e["eventId"] == self._pop_fork_root_event_id)
                events = events[: idx + 1]  # keep up to the fork root
            except StopIteration:
                # If we can't find the root, fall back to returning everything up to last-1
                events = events[:-1] if events else []

        items: List[TResponseInputItem] = []
        for ev in events:
            for p in ev.get("payload", []):
                conv = p.get("conversational")
                if not conv:
                    continue
                role = self._map_agentcore_role_to_openai(conv.get("role"))
                text = (conv.get("content") or {}).get("text") or ""
                if text == "":
                    continue
                # Responses API-friendly input item
                items.append({"role": role, "content": [_content_part_for_role(text, role)]})

        if limit is not None:
            items = items[-limit:]
        return items

    async def add_items(self, items: List[TResponseInputItem]) -> None:
        """Persist new items into AgentCore Memory.

        To support `pop_item()` granularity, we write **one Event per item**.
        If a pop was requested earlier, the first new event will transparently
        *fork* the conversation from the selected root, creating a fresh branch.
        """
        if not items:
            return

        # Determine if we need to create/continue a branch due to a prior pop()
        branch_for_first: Optional[Dict[str, str]] = None
        if self._pop_fork_root_event_id:
            self._current_branch = self._current_branch or self._gen_branch_name("fix")
            branch_for_first = {"rootEventId": self._pop_fork_root_event_id, "name": self._current_branch}
            self._pop_fork_root_event_id = None
            self._cleared = False

        # For subsequent events in the same add, we only need the branch name
        branch_for_rest: Optional[Dict[str, str]] = (
            ({"name": self._current_branch} if self._current_branch else None)
        )

        # Write each item as its own event
        for idx, item in enumerate(items):
            text, role = self._extract_text_and_role(item)
            if not text:
                continue  # skip empty items

            branch = branch_for_first if idx == 0 and branch_for_first else branch_for_rest

            await asyncio.to_thread(
                self._client.create_event,
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(text, role)],
                event_timestamp=dt.datetime.utcnow(),
                branch=branch,
            )

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item from this session (view).

        Implementation detail:
        - AgentCore doesn't support deleting an individual message within an event.
        - We store each message as a separate Event.
        - `pop_item()` sets an internal *fork root* to the previous Event. On the
          next `add_items()` call, we branch from that Event, effectively
          discarding the last Event from this Session's viewpoint.
        """
        events = await asyncio.to_thread(
            self._client.list_events,
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            branch_name=self._current_branch or None,
            include_parent_events=False,
            max_results=100,
            include_payload=True,
        )
        if not events:
            return None

        last = events[-1]
        prev = events[-2] if len(events) > 1 else None

        # Build return item from the last event's final conversational payload
        last_item: Optional[TResponseInputItem] = None
        for p in reversed(last.get("payload", [])):
            conv = p.get("conversational")
            if not conv:
                continue
            role = self._map_agentcore_role_to_openai(conv.get("role"))
            text = (conv.get("content") or {}).get("text") or ""
            if text:
                last_item = {"role": role, "content": [_content_part_for_role(text, role)]}
                break

        # Mark the fork root so the *next add* continues from `prev`
        self._pop_fork_root_event_id = prev["eventId"] if prev else None
        return last_item

    async def clear_session(self) -> None:
        """Clear this Session's view of history.

        This does **not** delete upstream events. We simply hide prior events
        until new ones are added (fresh conversation). If you need to hard-delete
        data, delete/recreate the Memory resource or rotate `session_id`.
        """
        self._cleared = True
        # We purposely do not change branches here; next add will resume on the
        # current branch (or main). If you need a hard reset, change session_id.

    # ------------- Optional helpers -------------
    def build_long_term_context(self, namespace: str, query: str, top_k: int = 3) -> List[TResponseInputItem]:
        """Retrieve top‑k long‑term memories and return them as a single system item.

        This lets you prepend semantic memories to your inputs when desired.
        """
        if "{sessionId}" in namespace:
            ns = namespace.replace("{sessionId}", self.session_id)
        else:
            ns = namespace

        memories: List[Dict[str, Any]] = self._client.retrieve_memories(
            memory_id=self.memory_id, namespace=ns, query=query, top_k=top_k
        ) or []

        if not memories:
            return []

        # Compact the memories into a short system item (you can customize this)
        lines = []
        for m in memories:
            content = (m.get("content") or {}).get("text")
            if content:
                lines.append(f"• {content}")
        text = "Relevant facts from long‑term memory:\n" + "\n".join(lines)
        # Use developer role to supply instructions-like context that won't be mistaken for a prior assistant output
        return [{"role": "developer", "content": [{"type": "input_text", "text": text}]}]

    # ------------- Internal utilities -------------
    @staticmethod
    def _gen_branch_name(prefix: str) -> str:
        return f"{prefix}-{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"

    @staticmethod
    def _extract_text_and_role(item: TResponseInputItem) -> Tuple[str, str]:
        """Return (text, AgentCoreRole) from a Responses input item, robust to missing roles.

        Heuristics:
        - If role missing -> infer from content parts:
            * any 'output_text' → assistant
            * else any 'input_text' → user
            * else → assistant
        - Extract text from the first part with type in {'output_text','input_text','text'}.
        """
        # --- Role
        role_raw: Optional[str] = None
        if isinstance(item, dict):
            role_raw = item.get("role")  # may be None
        role_lower = (role_raw or "").lower()

        # Content parts for role inference
        content = item.get("content") if isinstance(item, dict) else None
        parts = content if isinstance(content, list) else ([] if content is None else [content])
        inferred_role = None
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "output_text":
                inferred_role = "assistant"
                break
            if isinstance(part, dict) and part.get("type") == "input_text":
                inferred_role = inferred_role or "user"
        final_role = role_lower or (inferred_role or "assistant")

        # Map to AgentCore role
        if final_role == "user":
            ac_role = "USER"
        elif final_role == "assistant":
            ac_role = "ASSISTANT"
        else:
            # developer/system/tool/other → store as assistant context
            ac_role = "ASSISTANT"

        # --- Text
        text = ""
        if isinstance(item, dict):
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"output_text", "input_text", "text"}:
                        t = part.get("text")
                        if isinstance(t, str) and t:
                            text = t
                            break
        else:
            text = str(item)

        return text, ac_role

    @staticmethod
    def _map_agentcore_role_to_openai(ac_role: Optional[str]) -> str:
        r = (ac_role or "").upper()
        if r == "USER":
            return "user"
        if r == "ASSISTANT":
            return "assistant"
        if r == "TOOL":
            # Represent tool outputs as assistant-visible context
            return "assistant"
        return "assistant"
