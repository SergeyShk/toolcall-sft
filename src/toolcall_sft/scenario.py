"""The example scenario: send a payment to a saved payee.

Deliberately small. A tool-calling fine-tune is only as good as the agreement
between the prompt the model trains on and the prompt it is served with, so
both come from here — one system prompt, three tools, no per-request assembly,
nothing that has to be reconstructed at serving time.

Three tools is enough to exercise everything a tool-calling tune has to learn:
a lookup whose result decides what happens next (``find_payee``), a read that
only some branches need (``get_balance``), and a write that must not fire
without confirmation (``create_payment``).

Swap this file for your own prompt and tools to retarget the pipeline; nothing
else in the package knows what a payment is.
"""

from typing import Any

__all__ = ["SYSTEM_PROMPT", "TOOLS", "tool_names"]

SYSTEM_PROMPT = """You are a payment assistant for a small business banking app.
You help the customer send a payment to one of their saved payees.

Rules:
- Use the tools. Never invent payee details, balances, or payment ids.
- Ask for one missing thing at a time.
- A payment needs a payee, an amount, and a currency (default GBP).
- Repeat the payee name and the amount back to the customer and wait for a
  confirmation before calling create_payment.
- Keep replies to one or two short sentences.
- You can only send payments. Anything else — say so and stop."""

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "find_payee",
            "description": "Look up the customer's saved payees by name. Returns every match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full or partial payee name."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Current available balance of the customer's account.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_payment",
            "description": "Send a payment to a saved payee. Only call after the customer confirmed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payee_id": {"type": "string", "description": "Id from find_payee."},
                    "amount": {"type": "number", "description": "Amount in major units, e.g. 125.50."},
                    "currency": {"type": "string", "enum": ["GBP", "EUR", "USD"]},
                    "reference": {"type": "string", "description": "Payment reference; omit if none."},
                },
                "required": ["payee_id", "amount", "currency"],
            },
        },
    },
)


def tool_names() -> tuple[str, ...]:
    """Names of the tools the model is allowed to call."""
    return tuple(str(tool["function"]["name"]) for tool in TOOLS)
