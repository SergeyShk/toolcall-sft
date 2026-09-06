"""Deterministic PII anonymization for dialogues captured from a real system.

Three sources of sensitive terms:

- regex detectors for structured identifiers (emails, international phone
  numbers, IBANs, card numbers, 8-digit account numbers, UUIDs, ObjectIds).
  These are deliberately region-neutral. Locally-shaped identifiers — national
  ids, tax numbers, postal codes, domestic bank codes — differ per country and
  are yours to add; they are exactly what a generic list misses;
- values harvested from tool payloads under name-like keys — payee and
  account-holder names travel through tool results into replies, so the tool
  payloads know exactly which names to scrub. Harvesting is regex-based on the
  raw payload text: production tool results are not always clean JSON;
- a profile block in the system prompt (``- **Label**: value`` lines) —
  name labels become global name terms, date-of-birth and address values
  are replaced in place.

Financial fields (account number, IBAN, BIC) are additionally replaced by key
wherever ``"key": "value"`` appears in a payload, regardless of the value's
format — an account number written without separators still gets scrubbed.

Each text is scrubbed in a single pass: candidate spans are claimed over the
original text in priority order — key-driven financial values and protected
amounts first, then exact profile values (date of birth, address), then the
structured detectors, then harvested names — and a claimed span is never
rescanned. An email address containing a harvested name is therefore replaced
as one email, and a fake is never itself re-replaced.

Replacements are deterministic in (salt, kind, original), so the same value
maps to the same fake everywhere: an id passed from a tool result into a later
tool call stays consistent, and re-running the pass is reproducible.
Amounts are left untouched — they carry the scenario's semantics: values under
amount-like keys are protected explicitly, and digit runs adjacent to a
decimal dot are never treated as identifiers. A standalone 8-digit integer in
free text is still treated as an account number. Tool schemas are never
touched: their examples are static documentation and must stay byte-identical
to what the model will be served with.

This is a mechanical pass, not a guarantee: a free-text name that never
appears in a tool payload or the profile block survives it (an address inside
an attached invoice, for example). Review a sample before training and feed
known names in via ``extra_terms``.
"""

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import Dialogue, Message, Role, ToolCall

__all__ = ["AnonymizationReport", "Anonymizer"]

_FIRST_NAMES = (
    "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
    "Avery", "Quinn", "Rowan", "Harper", "Ellis", "Finley", "Sky", "Drew",
)  # fmt: skip
_LAST_NAMES = (
    "Walker", "Reed", "Hayes", "Brooks", "Marsh", "Bennett", "Cole", "Dale",
    "Ford", "Gray", "Hale", "Lane", "Nash", "Page", "Stone", "Wells",
)  # fmt: skip

_HARVEST_KEYS = frozenset(
    {
        "name", "first_name", "last_name", "full_name", "payee_name", "account_name",
        "account_holder", "account_holder_name", "beneficiary", "beneficiary_name",
        "company_name", "display_name", "legal_name", "trading_name", "customer_name",
        "payer_name", "payee", "recipient", "recipient_name", "holder_name", "business_name",
    }
)  # fmt: skip

_FINANCIAL_KEYS: Mapping[str, str] = {
    "account_number": "account_number",
    "iban": "iban",
    "bic_swift": "bic",
    "bic": "bic",
    "swift": "bic",
}

_SKIP_VALUES = frozenset({"-", "n/a", "none", "not set", "unknown", "null", "true", "false"})

# Keys whose numeric values are money, not identifiers — protected from the digit detectors.
_PROTECTED_KEYS = re.compile(r"amount|price|total|balance|fee|cents|minor_units")

_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("uuid", re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")),
    ("object_id", re.compile(r"\b[0-9a-f]{24}\b")),
    # International form only: a leading + and 8-15 digits, spaces or hyphens
    # between them. Dots are excluded on purpose so a decimal amount cannot look
    # like a phone number. Domestic formats are region-specific — add your own.
    ("phone", re.compile(r"\+(?:\d[\s\-]?){7,14}\d\b")),
    # Digit runs adjacent to a dot are decimal amounts, not card/account identifiers.
    ("card_number", re.compile(r"(?<![\d.])\d{13,19}(?![\d.])")),
    ("account_number", re.compile(r"(?<![\d.])\d{8}(?![\d.])")),
)

_KEY_VALUE = re.compile(r'"(?P<key>[a-z_]+)"(?P<sep>\s*:\s*)"(?P<value>[^"]{2,120})"')
_PROFILE_LINE = re.compile(r"(?m)^\s*-\s*\*\*(?P<label>[^*]+)\*\*\s*:\s*(?P<value>.+?)\s*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class AnonymizationReport:
    replacements: tuple[tuple[str, int], ...]

    @property
    def total(self) -> int:
        return sum(count for _kind, count in self.replacements)


class Anonymizer:
    __slots__ = ("__salt", "__extra_terms")

    def __init__(self, *, salt: str, extra_terms: Sequence[str] = ()) -> None:
        self.__salt = salt
        self.__extra_terms = tuple(term.strip() for term in extra_terms if len(term.strip()) >= 3)

    def anonymize_dialogue(self, dialogue: Dialogue) -> tuple[Dialogue, AnonymizationReport]:
        terms = self.__collect_terms(dialogue)
        counts: Counter[str] = Counter()
        messages = tuple(self.__anonymize_message(message, terms, counts) for message in dialogue.messages)
        anonymized = Dialogue(dialogue_id=dialogue.dialogue_id, messages=messages, tools=dialogue.tools)
        return anonymized, AnonymizationReport(replacements=tuple(sorted(counts.items())))

    def __collect_terms(self, dialogue: Dialogue) -> tuple[tuple[str, str], ...]:
        collected: dict[str, str] = {term.lower(): term for term in self.__extra_terms}
        kinds: dict[str, str] = {term.lower(): "name" for term in self.__extra_terms}
        for kind, term in _iter_dialogue_terms(dialogue):
            key = term.lower()
            if key not in collected:
                collected[key] = term
                kinds[key] = kind
        ordered = sorted(collected.values(), key=len, reverse=True)
        return tuple((kinds[term.lower()], term) for term in ordered)

    def __anonymize_message(self, message: Message, terms: Sequence[tuple[str, str]], counts: Counter[str]) -> Message:
        tool_calls = tuple(
            ToolCall(name=call.name, arguments=self.__scrub_json(call.arguments, terms, counts))
            for call in message.tool_calls
        )
        return Message(
            role=message.role,
            content=self.__scrub_text(message.content, terms, counts),
            tool_calls=tool_calls,
        )

    def __scrub_json(
        self, value: dict[str, Any], terms: Sequence[tuple[str, str]], counts: Counter[str]
    ) -> dict[str, Any]:
        scrubbed = self.__scrub_value(value, None, terms, counts)
        assert isinstance(scrubbed, dict)
        return scrubbed

    def __scrub_value(
        self, value: object, key: str | None, terms: Sequence[tuple[str, str]], counts: Counter[str]
    ) -> object:
        if isinstance(value, Mapping):
            return {str(k): self.__scrub_value(item, str(k), terms, counts) for k, item in value.items()}
        if isinstance(value, list):
            return [self.__scrub_value(item, key, terms, counts) for item in value]
        if isinstance(value, bool):
            return value
        financial_kind = _FINANCIAL_KEYS.get(key or "")
        if isinstance(value, int):
            if financial_kind is None:
                return value
            counts[financial_kind] += 1
            fake = self.__fake(financial_kind, str(value))
            return int(fake) if fake.isdigit() else fake
        if isinstance(value, str):
            if financial_kind is not None and _is_meaningful(value):
                counts[financial_kind] += 1
                return self.__fake(financial_kind, value)
            if key is not None and _PROTECTED_KEYS.search(key) and _is_numeric_like(value):
                return value
            return self.__scrub_text(value, terms, counts)
        return value

    def __scrub_text(self, text: str, terms: Sequence[tuple[str, str]], counts: Counter[str]) -> str:
        """Single pass over the original text: spans are collected by priority and never rescanned,
        so a fake can neither be re-replaced nor diverge from the same value faked elsewhere."""
        spans: list[tuple[int, int, str | None]] = []

        def claim(start: int, end: int, replacement: str | None) -> bool:
            if any(start < other_end and other_start < end for other_start, other_end, _ in spans):
                return False
            spans.append((start, end, replacement))
            return True

        for match in _KEY_VALUE.finditer(text):
            key = match.group("key")
            value = match.group("value")
            value_start, value_end = match.span("value")
            kind = _FINANCIAL_KEYS.get(key)
            if kind is not None and _is_meaningful(value):
                if claim(value_start, value_end, self.__fake(kind, value)):
                    counts[kind] += 1
            elif _PROTECTED_KEYS.search(key) and _is_numeric_like(value):
                claim(value_start, value_end, None)
        for kind, term in terms:
            if kind == "name":
                continue
            self.__claim_term(text, kind, term, claim, counts)
        for kind, pattern in _DETECTORS:
            for match in pattern.finditer(text):
                if claim(match.start(), match.end(), self.__fake(kind, match.group(0))):
                    counts[kind] += 1
        for kind, term in terms:
            if kind != "name":
                continue
            self.__claim_term(text, kind, term, claim, counts)

        if not spans:
            return text
        pieces: list[str] = []
        cursor = 0
        for start, end, replacement in sorted(spans, key=lambda span: span[0]):
            pieces.append(text[cursor:start])
            pieces.append(text[start:end] if replacement is None else replacement)
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)

    def __claim_term(
        self,
        text: str,
        kind: str,
        term: str,
        claim: Callable[[int, int, str | None], bool],
        counts: Counter[str],
    ) -> None:
        for match in re.finditer(re.escape(term), text, re.IGNORECASE):
            if claim(match.start(), match.end(), self.__fake(kind, term)):
                counts[kind] += 1

    def __fake(self, kind: str, original: str) -> str:
        digest = hashlib.sha256(f"{self.__salt}:{kind}:{original.lower()}".encode()).hexdigest()
        number = int(digest[:12], 16)
        first = _FIRST_NAMES[number % len(_FIRST_NAMES)]
        last = _LAST_NAMES[(number // 16) % len(_LAST_NAMES)]
        match kind:
            case "name":
                return f"{first} {last}"
            case "email":
                return f"{first.lower()}.{last.lower()}.{digest[:4]}@example.com"
            case "phone":
                # Country code 99 is unassigned, so this can never be a real number.
                return f"+99 900 {number % 1_000_000:06d}"
            case "account_number":
                return f"{number % 100_000_000:08d}"
            case "card_number":
                return f"4111{number % 10**12:012d}"
            case "iban":
                # XX is in the ISO 3166 user-assigned range: never a real country.
                return f"XX00TEST{number % 10**14:014d}"
            case "bic":
                return f"TESTXX{digest[:2].upper()}"
            case "date_of_birth":
                return f"{1 + number % 28} January 199{number % 10}"
            case "address":
                return f"{1 + number % 99} Example Street"
            case "uuid":
                return f"{digest[0:8]}-{digest[8:12]}-4{digest[13:16]}-a{digest[17:20]}-{digest[20:32]}"
            case "object_id":
                return digest[:24]
            case _:
                raise ValueError(f"unknown replacement kind {kind!r}")


def _iter_dialogue_terms(dialogue: Dialogue) -> Iterator[tuple[str, str]]:
    for message in dialogue.messages:
        for call in message.tool_calls:
            yield from _iter_argument_terms(call.arguments, None)
        if message.role is Role.TOOL:
            yield from _iter_payload_terms(message.content)
        if message.role is Role.SYSTEM:
            yield from _iter_profile_terms(message.content)


def _iter_payload_terms(content: str) -> Iterator[tuple[str, str]]:
    """Harvest name values from a tool payload by key, without requiring valid JSON."""
    for match in _KEY_VALUE.finditer(content):
        if match.group("key") in _HARVEST_KEYS and _is_meaningful(match.group("value")):
            yield "name", match.group("value")


def _iter_argument_terms(value: object, key: str | None) -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            yield from _iter_argument_terms(child, str(child_key))
    elif isinstance(value, list):
        for item in value:
            yield from _iter_argument_terms(item, key)
    elif isinstance(value, str) and key is not None and key.lower() in _HARVEST_KEYS and _is_meaningful(value):
        yield "name", value.strip()


def _iter_profile_terms(content: str) -> Iterator[tuple[str, str]]:
    """Harvest a profile block in the system prompt: name labels globally, DOB and address in place."""
    for match in _PROFILE_LINE.finditer(content):
        label = match.group("label").strip().lower()
        value = match.group("value").strip()
        if not _is_meaningful(value):
            continue
        if "date of birth" in label:
            yield "date_of_birth", value
        elif "address" in label:
            yield "address", value
        elif "name" in label:
            yield "name", value


def _is_meaningful(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 3 or stripped.lower() in _SKIP_VALUES:
        return False
    return not (stripped.startswith("<") and stripped.endswith(">"))


def _is_numeric_like(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and all(character in "0123456789.,- " for character in stripped)
