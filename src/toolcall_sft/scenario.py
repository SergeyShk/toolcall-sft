"""The example scenario: send a payment to a saved payee.

One system prompt (``system_prompt.txt``, read with surrounding whitespace
stripped) and three tools: a lookup whose result decides what happens next, a
write that must not fire without confirmation and can be declined, and an exit
for everything else. Training and serving must use both verbatim. Replace this
module and the prompt file to retarget the pipeline; nothing else in the package
knows what a payment is.
"""

from importlib.resources import files
from typing import Any

__all__ = ["SYSTEM_PROMPT", "TOOLS", "tool_names"]

SYSTEM_PROMPT = (files("toolcall_sft") / "system_prompt.txt").read_text(encoding="utf-8").strip()

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_payees",
            "description": (
                "Look up the customer's saved payees by name. Returns every match, so an "
                "empty list means the payee is not saved and more than one match has to be "
                "resolved with the customer before paying."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Full or partial payee name, as the customer said it.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_payment",
            "description": (
                "Send a payment to a saved payee. Call only after the customer has confirmed "
                "the payee and the amount. The response says whether the payment was actually "
                "sent — it can come back declined."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payee_id": {"type": "string", "description": "Id returned by get_payees."},
                    "amount": {"type": "number", "description": "Amount in major units, e.g. 125.50."},
                    "currency": {
                        "type": "string",
                        "enum": ["USD", "EUR", "CAD"],
                        "description": "ISO currency code. USD unless the customer said otherwise.",
                    },
                    "reference": {
                        "type": "string",
                        "description": "Payment reference shown to the payee. Omit if there is none.",
                    },
                },
                "required": ["payee_id", "amount", "currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": (
                "Hand the conversation to a human agent for anything other than sending a "
                "payment. Use it instead of answering a question you have no tool for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One sentence: what the customer asked for and why it needs a person.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
)


def tool_names() -> tuple[str, ...]:
    """Names of the tools the model is allowed to call."""
    return tuple(str(tool["function"]["name"]) for tool in TOOLS)
