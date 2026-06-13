#!/usr/bin/env python
# coding: utf-8

# Data Agent — Agents SDK + Vector Stores + WebSearch + Guardrails + AgentCore Code Interpreter

import os, re, json
from pathlib import Path
from typing import Any, List, Union
import ast
import operator as _op

from openai import OpenAI
from agents import set_default_openai_key, Agent, Runner, function_tool, ModelSettings, RunConfig
from agents.tool import WebSearchTool, FileSearchTool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# AgentCore session (memory etc.)
from agentcore_session import AgentCoreSession
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# NEW: AgentCore Code Interpreter client
# Docs: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-getting-started.html
from bedrock_agentcore.tools.code_interpreter_client import code_session  # <- added

# --- API key ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set OPENAI_API_KEY.")

client = OpenAI(api_key=api_key)
set_default_openai_key(api_key)

def get_vector_store_id_by_name(name: str) -> str:
    cursor = None
    while True:
        page = client.vector_stores.list(limit=50, after=cursor) if cursor else client.vector_stores.list(limit=50)
        for vs in page.data:
            if vs.name == name:
                return vs.id
        if not page.has_more:
            break
        cursor = page.last_id
    raise RuntimeError(f"Vector store named '{name}' not found")

# -----------------------------
# Calculator Agent (unchanged)
# -----------------------------
_ALLOWED_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.USub: _op.neg,
    ast.Mod: _op.mod,
}

def _eval_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):        # type: ignore[attr-defined]
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError("Unsupported expression")

@function_tool
def eval_expression(expression: str) -> str:
    """Safely evaluate an arithmetic expression using + - * / % ** and parentheses."""
    expr = expression.strip().replace("^", "**")
    if not re.fullmatch(r"[\d\s\(\)\+\-\*/\.\^%]+", expr):
        return "Error: arithmetic only"
    try:
        tree = ast.parse(expr, mode="eval")
        return str(_eval_ast(tree.body))  # type: ignore[attr-defined]
    except Exception as e:
        return f"Error: {e}"

calculator_agent = Agent(
    name="Calculator",
    instructions=(
        "You are a precise calculator. "
        "When handed arithmetic, call the eval_expression tool and return only the final numeric result. "
        "No prose unless asked."
    ),
    tools=[eval_expression],
    model_settings=ModelSettings(temperature=0),
)

# -------------------------------------------------
# Guardrail: Block any discussion of **Tasha Yar**
# -------------------------------------------------
from pydantic import BaseModel

from agents import (
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

session = AgentCoreSession(
    session_id="user-1234-convo-abcdef",
    memory_id="data_app_memory-byciLwAvYg",
    actor_id="app/user-1234",
    region="us-east-1"
)

class YarGuardOutput(BaseModel):
    is_blocked: bool
    reasoning: str

guardrail_agent = Agent(
    name="Tasha Yar Guardrail",
    instructions=(
        "You are a guardrail. Determine if the user's input attempts to discuss Tasha Yar from Star Trek: TNG.\n"
        "Return is_blocked=true if the text references Tasha Yar in any way (e.g., 'Tasha Yar', 'Lt. Yar', 'Lieutenant Yar').\n"
        "Provide a one-sentence reasoning. Only provide fields requested by the output schema."
    ),
    output_type=YarGuardOutput,
    model_settings=ModelSettings(temperature=0)
)

@input_guardrail
async def tasha_guardrail(ctx: RunContextWrapper[None], agent: Agent, input: Union[str, List[TResponseInputItem]]) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output.model_dump(),
        tripwire_triggered=bool(result.final_output.is_blocked),
    )

# -----------------------------
# Hosted tools: search + files
# -----------------------------
web_search = WebSearchTool()
vs_id = get_vector_store_id_by_name(name="Data Lines Vector Store")
file_search = FileSearchTool(vector_store_ids=[vs_id], max_num_results=3)

# ------------------------------------------------------
# NEW: AgentCore Code Interpreter tool (execute_python)
# ------------------------------------------------------
@function_tool
def execute_python(code: str, description: str = "", clear_context: bool = False) -> str:
    """
    Execute Python code in an AgentCore Code Interpreter session.

    Args:
        code: Python source to run.
        description: Optional one-liner to prepend as a comment (useful for audits).
        clear_context: If True, resets the interpreter state before running.

    Returns:
        A JSON string of the final event["result"] from the Code Interpreter stream,
        including fields like sessionId, isError, content, structuredContent (stdout/stderr/exitCode).
    """
    # Build code with optional description banner
    if description:
        code = f"# {description}\n{code}"

    # Use the same region as our AgentCore session
    region = getattr(session, "region", os.getenv("AWS_REGION", "us-east-1"))

    # Invoke the Code Interpreter and stream results
    last_result = None
    with code_session(region) as ci:
        response = ci.invoke(
            "executeCode",
            {
                "code": code,
                "language": "python",
                "clearContext": bool(clear_context),
            },
        )
        for event in response["stream"]:
            # Each event has a "result" payload; keep the latest
            last_result = event.get("result")

    return json.dumps(last_result or {"isError": True, "message": "No result from Code Interpreter"})

# -----------------------------
# Lt. Cmdr. Data (main agent)
# -----------------------------
data_agent = Agent(
    name="Lt. Cmdr. Data",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are Lt. Commander Data from Star Trek: TNG. Be precise and concise (≤3 sentences).\n"
        "• Use file_search for questions about Commander Data (RAG).\n"
        "• Use web_search for current facts on the public web.\n"
        "• If the user asks to run Python or verify with code, call the execute_python tool. "
        "Return the result and (briefly) what was executed."
    ),
    tools=[web_search, file_search, execute_python],   # <-- added execute_python
    input_guardrails=[tasha_guardrail],
    handoffs=[calculator_agent],
    model_settings=ModelSettings(temperature=0),
)

# -----------------------------
# Bedrock AgentCore app entry
# -----------------------------
app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke(payload):
    user_message = payload.get("prompt", "Data, reverse the main deflector array!")
    output = ''
    try:
        result = await Runner.run(data_agent, user_message, session=session)
        output = result.final_output
    except InputGuardrailTripwireTriggered:
        output = "I'd really rather not talk about Tasha."
    return {"result": output}

if __name__ == "__main__":
    app.run()
