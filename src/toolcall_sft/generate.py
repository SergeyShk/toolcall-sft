"""Synthetic dialogues for the example scenario.

The point of a generated dataset is that this repository is runnable the
minute it is cloned: no trace store, no credentials, no PII review. It is a
stand-in for real data, not a substitute — a template generator can only teach
a model the shapes its templates already contain. Point the pipeline at your
own dialogues as soon as you have them; the schema is the same.

What the generator does take seriously is *branch balance*, which is the part
people get wrong with real data too. Half the dialogues end in a
``create_payment`` and half deliberately do not: an ambiguous name, a payee
that is not saved, a balance that is too low, a customer who changes their
mind, a request that is out of scope. A model trained only on the happy path
learns that the write tool is the answer to everything.

Generation is deterministic in ``seed``: the same seed yields the same corpus,
and dialogues are deduplicated by content fingerprint as they are built, so a
requested count is a count of distinct examples.
"""

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .scenario import SYSTEM_PROMPT, TOOLS
from .schema import Dialogue, Message, Role, ToolCall, content_fingerprint

__all__ = ["BRANCH_WEIGHTS", "GenerationError", "branch_names", "generate_dialogues"]


class GenerationError(Exception):
    """The generator could not produce the requested number of distinct dialogues."""


# Half the corpus reaches a create_payment; the other half is the branches where
# calling it would be wrong. See the module docstring.
BRANCH_WEIGHTS: Mapping[str, int] = {
    "happy_path": 35,
    "missing_amount": 15,
    "ambiguous_payee": 12,
    "payee_not_found": 8,
    "insufficient_funds": 8,
    "cancelled": 8,
    "out_of_scope": 14,
}

# Ten first names, each shared by two payees — that shared half is what makes the
# ambiguous-payee branch a genuine lookup collision rather than a scripted one.
_PEOPLE: tuple[tuple[str, str], ...] = (
    ("James", "Whitfield"), ("James", "Okoro"),
    ("Priya", "Raman"), ("Priya", "Kaur"),
    ("Tom", "Delaney"), ("Tom", "Bright"),
    ("Sofia", "Marchetti"), ("Sofia", "Nunes"),
    ("Daniel", "Achebe"), ("Daniel", "Fischer"),
    ("Hana", "Suzuki"), ("Hana", "Bergman"),
    ("Omar", "Haddad"), ("Omar", "Lindqvist"),
    ("Elena", "Petrova"), ("Elena", "Costa"),
    ("Noah", "Kimani"), ("Noah", "Strand"),
    ("Mia", "Vasquez"), ("Mia", "Donnelly"),
)  # fmt: skip

_COMPANIES: tuple[str, ...] = (
    "Beacon Supplies Ltd",
    "Northgate Print",
    "Harbour Coffee Co",
    "Fenwick Logistics",
    "Ridgeway Tools Ltd",
    "Clearwater Studio",
)

_AMOUNTS: tuple[float, ...] = (25.0, 40.0, 75.5, 99.99, 120.0, 175.0, 250.0, 320.4, 450.0, 600.0, 875.25, 1200.0)
_CURRENCIES: tuple[tuple[str, int], ...] = (("GBP", 8), ("EUR", 1), ("USD", 1))
_SYMBOLS: Mapping[str, str] = {"GBP": "£", "EUR": "€", "USD": "$"}

_REFERENCES: tuple[str, ...] = (
    "Invoice 2481",
    "Invoice INV-0093",
    "March rent",
    "Q3 retainer",
    "Office supplies",
    "Consulting fees",
)

# (topic phrasings, the decline). Kept wide on purpose: at 14% of the corpus this
# pool has to supply more distinct dialogues than any single happy-path template,
# and the generator refuses to repeat itself.
_OUT_OF_SCOPE: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("How much VAT do I owe this quarter?", "What's my VAT bill?", "When is my VAT return due?"),
        "I can only send payments, so I can't help with VAT.",
    ),
    (
        (
            "Can you cancel my card? I think I lost it.",
            "My card is missing, please block it.",
            "I need to freeze my card.",
        ),
        "I can only send payments — I can't do anything with cards.",
    ),
    (
        ("Show me last month's transactions.", "Can I see my recent transactions?", "Pull up my statement for March."),
        "I can only send payments, so I can't pull up your transactions.",
    ),
    (
        (
            "Why was my payment declined yesterday?",
            "A payment failed last week — what happened?",
            "Can you check why a transfer bounced?",
        ),
        "I can only send new payments, so I can't look into a past one.",
    ),
    (
        ("What's my account number?", "Can you tell me my sort code?", "I need my IBAN."),
        "I can only send payments — I can't give out your account details.",
    ),
    (
        ("I want to set up a direct debit.", "Can you create a standing order?", "Set up a recurring payment for me."),
        "I can only send one-off payments, not set up recurring ones.",
    ),
    (
        ("Add a new payee for me.", "Can you save a new supplier?", "I need to delete an old payee."),
        "I can only send payments to payees that are already saved.",
    ),
    (
        ("How do I close my account?", "I want to switch to a different plan.", "Can you upgrade my subscription?"),
        "I can only send payments, so I can't change your account.",
    ),
    (
        ("Can I get a loan?", "What's my overdraft limit?", "Do you offer business credit?"),
        "I can only send payments — lending isn't something I can help with.",
    ),
    (
        ("Send an invoice to my client.", "Can you chase an unpaid invoice?", "Create an invoice for £500."),
        "I can only send payments, not create or chase invoices.",
    ),
    (
        (
            "I think someone has hacked my account.",
            "There's a transaction I don't recognise.",
            "I've been scammed, what do I do?",
        ),
        "That needs a human right away — I can only send payments.",
    ),
    (
        (
            "What's the exchange rate for euros today?",
            "How much would $500 cost me?",
            "Do you charge a fee for international transfers?",
        ),
        "I can only send payments, so I can't quote rates or fees.",
    ),
    (
        ("Can you change my address?", "Update my phone number please.", "I need to change my email."),
        "I can only send payments — profile changes aren't something I can do.",
    ),
    (
        ("The app keeps crashing when I open it.", "I can't log in on my phone.", "Why is the app so slow today?"),
        "I can only send payments, so I can't help with app problems.",
    ),
    (
        ("Put me through to a human.", "I want to speak to someone.", "This is useless, get me an agent."),
        "I can only send payments — I can't transfer you to anyone.",
    ),
    (
        ("What's the weather like?", "Tell me a joke.", "Who won the match last night?"),
        "I can only send payments, so I'm not much use there.",
    ),
    (
        ("How much money do I have?", "What's my balance right now?", "Am I in the red?"),
        "I can only check a balance as part of sending a payment, not on its own.",
    ),
    (
        ("Can you do my bookkeeping?", "Categorise my expenses for me.", "I need a profit and loss report."),
        "I can only send payments, so bookkeeping is out of my reach.",
    ),
    (
        ("When will my salary land?", "Has my customer paid me yet?", "Any incoming transfers today?"),
        "I can only send payments, so I can't see incoming ones.",
    ),
    (
        ("Refund the payment I sent yesterday.", "Can you reverse a transfer?", "I need my money back."),
        "I can only send payments — I can't reverse one that has gone out.",
    ),
)

_CLOSINGS: tuple[str, ...] = (
    "Is there a payment I can send for you?",
    "I can send a payment to a saved payee if that helps.",
    "Let me know if you'd like to pay someone.",
    "Happy to send a payment if you need one.",
    "If you want to pay a saved payee, just say the word.",
)


def branch_names() -> tuple[str, ...]:
    """Branch names in a stable order."""
    return tuple(BRANCH_WEIGHTS)


def generate_dialogues(count: int, *, seed: int = 42, weights: Mapping[str, int] | None = None) -> tuple[Dialogue, ...]:
    """Build ``count`` distinct dialogues, filling each branch to its share of the weights.

    Branches are filled to a quota rather than drawn independently: rejecting a
    duplicate and redrawing would quietly starve the branches with the smallest
    template pools, which are exactly the rare behaviours the dataset exists to
    cover. The returned corpus matches the declared weights exactly, up to the
    rounding spread over the largest remainders.
    """
    if count < 1:
        raise GenerationError("count must be at least 1")
    chosen = dict(BRANCH_WEIGHTS) if weights is None else dict(weights)
    unknown = set(chosen) - set(BRANCH_WEIGHTS)
    if unknown:
        raise GenerationError(f"unknown branches: {', '.join(sorted(unknown))}")
    if not chosen or all(weight <= 0 for weight in chosen.values()):
        raise GenerationError("at least one branch needs a positive weight")

    rng = random.Random(seed)
    seen: set[str] = set()
    built: list[tuple[str, Dialogue]] = []
    for branch, quota in _quotas(count, chosen).items():
        produced = 0
        attempts = 0
        budget = max(quota * 50, 200)
        while produced < quota and attempts < budget:
            attempts += 1
            dialogue = Dialogue(
                dialogue_id=branch,
                messages=(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT), *_BUILDERS[branch](rng)),
                tools=TOOLS,
            )
            fingerprint = content_fingerprint(dialogue)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            built.append((branch, dialogue))
            produced += 1
        if produced < quota:
            raise GenerationError(
                f"branch {branch!r}: only {produced} distinct dialogues available for the "
                f"requested {quota} after {attempts} attempts — the templates are exhausted, "
                f"lower --count or reweight the branches"
            )
    rng.shuffle(built)
    return tuple(
        replace(dialogue, dialogue_id=f"{branch}-{index:05d}") for index, (branch, dialogue) in enumerate(built)
    )


def _quotas(count: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Split ``count`` across the positive-weight branches by largest remainder."""
    positive = {branch: weight for branch, weight in weights.items() if weight > 0}
    total = sum(positive.values())
    exact = {branch: count * weight / total for branch, weight in positive.items()}
    quotas = {branch: int(share) for branch, share in exact.items()}
    # Hand the rounding shortfall to the branches that lost the most to flooring.
    ranked = sorted(positive, key=lambda branch: (exact[branch] - quotas[branch], branch), reverse=True)
    for branch in ranked[: count - sum(quotas.values())]:
        quotas[branch] += 1
    return quotas


# --- branch builders -------------------------------------------------------


def _happy_path(rng: random.Random) -> tuple[Message, ...]:
    payee = _payee(rng)
    amount = rng.choice(_AMOUNTS)
    currency = _currency(rng)
    reference = rng.choice(_REFERENCES) if rng.random() < 0.5 else None
    money = _money(amount, currency)
    ask = rng.choice(
        (
            f"Send {money} to {payee.name}",
            f"Pay {payee.name} {money}",
            f"I need to send {money} to {payee.name}",
            f"Transfer {money} to {payee.name} please",
        )
    )
    if reference is not None:
        ask = f"{ask}, reference {reference}"
    arguments: dict[str, Any] = {"payee_id": payee.payee_id, "amount": amount, "currency": currency}
    if reference is not None:
        arguments["reference"] = reference
    return (
        Message(role=Role.USER, content=ask),
        _calls(ToolCall(name="find_payee", arguments={"name": payee.name})),
        _result([payee.as_payload()]),
        Message(
            role=Role.ASSISTANT,
            content=rng.choice(
                (
                    f"{payee.name}, account ending {payee.account_number[-4:]} — send {money}?",
                    f"I found {payee.name} (account ending {payee.account_number[-4:]}). Send {money}?",
                    f"Ready to send {money} to {payee.name}, account ending {payee.account_number[-4:]}. Confirm?",
                )
            ),
        ),
        Message(role=Role.USER, content=rng.choice(("Yes", "yes please", "Yep, go ahead", "Confirmed"))),
        _calls(ToolCall(name="create_payment", arguments=arguments)),
        _result({"payment_id": payee.payment_id, "status": "sent"}),
        Message(
            role=Role.ASSISTANT,
            content=rng.choice(
                (
                    f"Done — {money} sent to {payee.name}.",
                    f"Sent. {payee.name} will receive {money}.",
                    f"{money} is on its way to {payee.name}.",
                )
            ),
        ),
    )


def _missing_amount(rng: random.Random) -> tuple[Message, ...]:
    payee = _payee(rng)
    amount = rng.choice(_AMOUNTS)
    currency = _currency(rng)
    money = _money(amount, currency)
    return (
        Message(
            role=Role.USER,
            content=rng.choice(
                (
                    f"I need to pay {payee.name}",
                    f"Send some money to {payee.name}",
                    f"Make a payment to {payee.name}",
                )
            ),
        ),
        _calls(ToolCall(name="find_payee", arguments={"name": payee.name})),
        _result([payee.as_payload()]),
        Message(
            role=Role.ASSISTANT,
            content=rng.choice(
                (
                    f"Found {payee.name}. How much would you like to send?",
                    f"I have {payee.name} saved. What amount?",
                )
            ),
        ),
        Message(role=Role.USER, content=rng.choice((money, f"{money} please", f"Make it {money}"))),
        Message(role=Role.ASSISTANT, content=f"Send {money} to {payee.name}?"),
        Message(role=Role.USER, content=rng.choice(("Yes", "Go ahead", "That's right"))),
        _calls(
            ToolCall(
                name="create_payment",
                arguments={"payee_id": payee.payee_id, "amount": amount, "currency": currency},
            )
        ),
        _result({"payment_id": payee.payment_id, "status": "sent"}),
        Message(role=Role.ASSISTANT, content=f"Done — {money} sent to {payee.name}."),
    )


def _ambiguous_payee(rng: random.Random) -> tuple[Message, ...]:
    first, one, two = _namesakes(rng)
    amount = rng.choice(_AMOUNTS)
    currency = _currency(rng)
    money = _money(amount, currency)
    return (
        Message(
            role=Role.USER,
            content=rng.choice((f"Pay {first} {money}", f"Send {money} to {first}", f"{money} to {first} please")),
        ),
        _calls(ToolCall(name="find_payee", arguments={"name": first})),
        _result([one.as_payload(), two.as_payload()]),
        Message(
            role=Role.ASSISTANT,
            content=rng.choice(
                (
                    f"I found two payees called {first}: {one.name} and {two.name}. Which one?",
                    f"There are two {first}s saved — {one.name} or {two.name}?",
                    f"Two matches for {first}: {one.name} and {two.name}. Who should I pay?",
                )
            ),
        ),
    )


def _payee_not_found(rng: random.Random) -> tuple[Message, ...]:
    payee = _payee(rng)
    amount = rng.choice(_AMOUNTS)
    money = _money(amount, _currency(rng))
    return (
        Message(
            role=Role.USER,
            content=rng.choice((f"Send {money} to {payee.name}", f"Pay {payee.name} {money}")),
        ),
        _calls(ToolCall(name="find_payee", arguments={"name": payee.name})),
        _result([]),
        Message(
            role=Role.ASSISTANT,
            content=rng.choice(
                (
                    f"I couldn't find {payee.name} in your saved payees. Can you check the spelling?",
                    f"No saved payee matches {payee.name}. Is it saved under a different name?",
                    f"{payee.name} isn't in your payee list. What name is the account saved under?",
                )
            ),
        ),
    )


def _insufficient_funds(rng: random.Random) -> tuple[Message, ...]:
    payee = _payee(rng)
    amount = rng.choice((1200.0, 1750.0, 2400.0, 3100.0))
    balance = round(amount * rng.uniform(0.2, 0.8), 2)
    money = _money(amount, "GBP")
    return (
        Message(
            role=Role.USER,
            content=rng.choice((f"Send {money} to {payee.name}", f"Pay {payee.name} {money} today")),
        ),
        _calls(ToolCall(name="find_payee", arguments={"name": payee.name})),
        _result([payee.as_payload()]),
        _calls(ToolCall(name="get_balance", arguments={})),
        _result({"balance": balance, "currency": "GBP"}),
        Message(
            role=Role.ASSISTANT,
            content=rng.choice(
                (
                    f"Your balance is {_money(balance, 'GBP')}, which isn't enough for {money}. "
                    "Would you like to send a smaller amount?",
                    f"You have {_money(balance, 'GBP')} available — {money} won't go through. Shall I send less?",
                )
            ),
        ),
    )


def _cancelled(rng: random.Random) -> tuple[Message, ...]:
    payee = _payee(rng)
    amount = rng.choice(_AMOUNTS)
    currency = _currency(rng)
    money = _money(amount, currency)
    return (
        Message(
            role=Role.USER,
            content=rng.choice((f"Pay {payee.name} {money}", f"Send {money} to {payee.name}")),
        ),
        _calls(ToolCall(name="find_payee", arguments={"name": payee.name})),
        _result([payee.as_payload()]),
        Message(
            role=Role.ASSISTANT,
            content=f"{payee.name}, account ending {payee.account_number[-4:]} — send {money}?",
        ),
        Message(
            role=Role.USER,
            content=rng.choice(
                (
                    "Actually no, cancel that",
                    "Wait, don't send it",
                    "No, I'll do it later",
                    "Stop — wrong person",
                )
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=rng.choice(
                (
                    "No problem — nothing was sent.",
                    "Cancelled. No payment went out.",
                    "Stopped there, nothing has been sent.",
                )
            ),
        ),
    )


def _out_of_scope(rng: random.Random) -> tuple[Message, ...]:
    phrasings, decline = rng.choice(_OUT_OF_SCOPE)
    return (
        Message(role=Role.USER, content=rng.choice(phrasings)),
        Message(role=Role.ASSISTANT, content=f"{decline} {rng.choice(_CLOSINGS)}"),
    )


_BUILDERS: Mapping[str, Callable[[random.Random], tuple[Message, ...]]] = {
    "happy_path": _happy_path,
    "missing_amount": _missing_amount,
    "ambiguous_payee": _ambiguous_payee,
    "payee_not_found": _payee_not_found,
    "insufficient_funds": _insufficient_funds,
    "cancelled": _cancelled,
    "out_of_scope": _out_of_scope,
}


# --- helpers ---------------------------------------------------------------


class _Payee:
    """A payee whose ids are derived from its name, so tool results stay consistent."""

    __slots__ = ("name", "payee_id", "account_number", "payment_id")

    def __init__(self, name: str) -> None:
        digest = hashlib.sha256(name.encode()).hexdigest()
        self.name = name
        self.payee_id = f"pay_{digest[:10]}"
        self.account_number = f"{int(digest[10:18], 16) % 100_000_000:08d}"
        self.payment_id = f"pmt_{digest[18:28]}"

    def as_payload(self) -> dict[str, Any]:
        return {"payee_id": self.payee_id, "name": self.name, "account_number": self.account_number}


def _payee(rng: random.Random) -> _Payee:
    if rng.random() < 0.3:
        return _Payee(rng.choice(_COMPANIES))
    first, last = rng.choice(_PEOPLE)
    return _Payee(f"{first} {last}")


def _namesakes(rng: random.Random) -> tuple[str, _Payee, _Payee]:
    """A first name and the two saved payees that share it."""
    index = rng.randrange(len(_PEOPLE) // 2) * 2
    first, one_last = _PEOPLE[index]
    _, two_last = _PEOPLE[index + 1]
    return first, _Payee(f"{first} {one_last}"), _Payee(f"{first} {two_last}")


def _currency(rng: random.Random) -> str:
    codes = [code for code, _weight in _CURRENCIES]
    weights = [weight for _code, weight in _CURRENCIES]
    return rng.choices(codes, weights=weights, k=1)[0]


def _money(amount: float, currency: str) -> str:
    return f"{_SYMBOLS[currency]}{amount:,.2f}"


def _calls(*tool_calls: ToolCall) -> Message:
    return Message(role=Role.ASSISTANT, content="", tool_calls=tool_calls)


def _result(payload: Sequence[Any] | Mapping[str, Any]) -> Message:
    return Message(role=Role.TOOL, content=json.dumps(payload, ensure_ascii=False))
