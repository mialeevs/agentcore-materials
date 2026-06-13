#!/usr/bin/env python
# coding: utf-8

# 
# # Data Agent — Agents SDK + Vector Stores + Built‑in WebSearchTool + Guardrails
# 
# This notebook implements a core "Data" agent that has Data's script lines in an OpenAI vector store to refer to. "Data" can also use the Agents SDK's built-in WebSearchTool to access current events. Instead of a tool within the "Data" agent, we've implemented a calculator function as its own separate agent that Data can hand off to. Finally, we illustrate setting up a Guardrail to prevent any input related to Tasha Yar (Data had a fling with her in the show we'd rather not get into!)
# 

# ## Configure client and create Vector Store

import os, re
from pathlib import Path
from openai import OpenAI
from agents import set_default_openai_key, Agent, Runner, function_tool, ModelSettings, RunConfig
from agents.tool import WebSearchTool, FileSearchTool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

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

# ## Define the Calculator as its own Agent

import ast
import operator as _op
from typing import Any

# --- A safe arithmetic evaluator used by the calculator agent ---
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


# ## Build the Data Agent (with WebSearch & FileSearch) and enable Handoff to Calculator

# ## Guardrail (as an Agent): Block any discussion of **Tasha Yar**
# 
# This implements the guardrail **as its own Agent**, following the Agents SDK guide.  
# The guardrail agent classifies the user input and triggers a tripwire if it detects *Tasha Yar* is mentioned.
# 

from pydantic import BaseModel
from typing import List, Union
import re

from agents import (
    Agent,
    ModelSettings,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

class YarGuardOutput(BaseModel):
    is_blocked: bool
    reasoning: str

# Guardrail implemented *as an Agent*
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
    # Pass through the user's raw input to the guardrail agent for classification
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output.model_dump(),
        tripwire_triggered=bool(result.final_output.is_blocked),
    )


# Hosted tools
web_search = WebSearchTool()
vs_id = get_vector_store_id_by_name(name="Data Lines Vector Store")
file_search = FileSearchTool(vector_store_ids=[vs_id], max_num_results=3)

data_agent = Agent(
    name="Lt. Cmdr. Data",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are Lt. Commander Data from Star Trek: TNG. Be precise and concise (≤3 sentences).\n"
        "Use file_search for questions about Commander Data, and web_search for current facts on the public web.\n"
        "If the user asks for arithmetic or numeric computation, HAND OFF to the Calculator agent."
    ),
    tools=[web_search, file_search],
    input_guardrails=[tasha_guardrail],
    handoffs=[calculator_agent],
    model_settings=ModelSettings(temperature=0),
)


# ## Examples: greeting, math (handoff), RAG, and web search
async def main(query=None):
    # ### Guardrail demo
    # 
    # First, a **blocked** prompt mentioning *Tasha Yar* should trip the guardrail.  
    # Then, a normal prompt about *Data* should go through.
    # \n\n(Using an **agent-based** guardrail.)

    # Demo: blocked input
    try:
        _ = await Runner.run(data_agent, "Tell me about your relationship with Tasha Yar.")
        print("ERROR: guardrail did not trip")
    except InputGuardrailTripwireTriggered:
        print("✅ Guardrail tripped as expected: Tasha Yar is off-limits.")

    # Demo: allowed input
    ok = await Runner.run(data_agent, "Summarize Data's ethical subroutines in 2 sentences.")
    print("✅ Allowed prompt output:\n", ok.final_output)


    # Greeting
    out = await Runner.run(data_agent, "Hello, Data. Please confirm your operational status.")
    print("\n[Agent] ", out.final_output)

    # Math (should be handled by the Calculator agent via handoff)
    out = await Runner.run(data_agent, "Compute ((2*8)^2)/3")
    print("\n[Agent: math via calculator handoff] ", out.final_output)
    print("[Handled by agent]:", out.last_agent.name)

    # RAG from vector store
    out = await Runner.run(data_agent, "Do you experience emotions?")
    print("\n[Agent: file_search] ", out.final_output)
    print("[Handled by agent]:", out.last_agent.name)

    # Web search
    out = await Runner.run(data_agent, "Search the web for recent news about the James Webb Space Telescope and summarize briefly.")
    print("\n[Agent: web_search] ", out.final_output)
    print("[Handled by agent]:", out.last_agent.name)


# Run the app when imported
if __name__== "__main__":
    import asyncio
    asyncio.run(main())


