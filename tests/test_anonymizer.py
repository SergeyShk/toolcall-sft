from toolcall_sft import Anonymizer, Dialogue, Message, Role, ToolCall


def _payment_dialogue() -> Dialogue:
    return Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.SYSTEM, content="Customer email: john.smith@acme.example, phone +49 151 2345678."),
            Message(role=Role.USER, content="Pay $100 to John Smith, IBAN DE89370400440532013000, account 12345678"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name="get_payees", arguments={"payee_name": "John Smith"}),),
            ),
            Message(role=Role.TOOL, content='[{"payee_id": "p1", "name": "John Smith"}]'),
            Message(role=Role.ASSISTANT, content="I found JOHN SMITH — confirm $100?"),
        ),
    )


def test_anonymize_replaces_structured_identifiers() -> None:
    anonymized, report = Anonymizer(salt="v1").anonymize_dialogue(_payment_dialogue())

    system = anonymized.messages[0].content
    user = anonymized.messages[1].content
    assert "john.smith@acme.example" not in system
    assert "@example.com" in system
    assert "+49 151 2345678" not in system
    assert "DE89370400440532013000" not in user
    assert "12345678" not in user
    assert "$100" in user  # amounts carry the scenario's meaning and are kept
    assert report.total > 0


def test_anonymize_harvests_names_from_tool_payloads() -> None:
    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(_payment_dialogue())

    final = anonymized.messages[-1].content
    tool_result = anonymized.messages[3].content
    arguments = anonymized.messages[2].tool_calls[0].arguments
    assert "JOHN SMITH" not in final.upper() or "John Smith" not in final
    assert "John Smith" not in tool_result
    assert arguments["payee_name"] != "John Smith"
    # The same original maps to the same fake in every location, case-insensitively.
    fake = arguments["payee_name"]
    assert fake in tool_result
    assert fake in final


def test_anonymize_is_deterministic_and_salt_sensitive() -> None:
    dialogue = _payment_dialogue()

    first, _ = Anonymizer(salt="v1").anonymize_dialogue(dialogue)
    second, _ = Anonymizer(salt="v1").anonymize_dialogue(dialogue)
    other_salt, _ = Anonymizer(salt="v2").anonymize_dialogue(dialogue)

    assert first == second
    assert first != other_salt


def test_anonymize_extra_terms_are_redacted() -> None:
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="This is Acme Widgets speaking"),
            Message(role=Role.ASSISTANT, content="Hello Acme Widgets!"),
        ),
    )

    anonymized, report = Anonymizer(salt="v1", extra_terms=("Acme Widgets",)).anonymize_dialogue(dialogue)

    assert "Acme Widgets" not in anonymized.messages[0].content
    assert "Acme Widgets" not in anonymized.messages[1].content
    assert report.total == 2


def test_anonymize_harvests_from_payload_with_trailing_garbage() -> None:
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="Pay this invoice"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name="parse_file_invoice", arguments={"file_id": "f1"}),),
            ),
            Message(
                role=Role.TOOL,
                content=(
                    '{\n  "account_name": "Starbridge Group",\n  "account_number": "40062700"\n}\nBanner: check limits'
                ),
            ),
            Message(role=Role.ASSISTANT, content="Draft for Starbridge Group created"),
        ),
    )

    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    assert "Starbridge" not in anonymized.messages[2].content
    assert "Starbridge" not in anonymized.messages[3].content
    assert "40062700" not in anonymized.messages[2].content


def test_anonymize_replaces_financial_fields_by_key_in_arguments() -> None:
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="Pay"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(
                    ToolCall(
                        name="create_payment",
                        arguments={
                            "iban": "DE89370400440532013000",
                            "account_number": "81544756",
                            "amount": "100.00",
                        },
                    ),
                ),
            ),
            Message(role=Role.TOOL, content='{"status": "ok"}'),
            Message(role=Role.ASSISTANT, content="Done"),
        ),
    )

    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    arguments = anonymized.messages[1].tool_calls[0].arguments
    assert arguments["iban"] != "DE89370400440532013000"
    assert arguments["account_number"] != "81544756"
    assert arguments["amount"] == "100.00"


def test_anonymize_scrubs_profile_block() -> None:
    system = (
        "You are a payment assistant.\n"
        "Customer Profile\n"
        "- **Full Name**: Ivan Petrov\n"
        "- **First Name**: Ivan\n"
        "- **Date of Birth**: 12 March 1985\n"
        "- **Registered Address**: 5 Real Street, Leeds, LS1 4AB\n"
        "- **Preferred Name**: -\n"
    )
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content="Hi, this is Ivan Petrov"),
            Message(role=Role.ASSISTANT, content="Hello Ivan!"),
        ),
    )

    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    scrubbed_system = anonymized.messages[0].content
    assert "Ivan" not in scrubbed_system
    assert "12 March 1985" not in scrubbed_system
    assert "Real Street" not in scrubbed_system
    assert "LS1 4AB" not in scrubbed_system
    assert "Ivan" not in anonymized.messages[1].content
    assert "Ivan" not in anonymized.messages[2].content


def test_anonymize_detects_international_phone_and_iban() -> None:
    """The structured detectors are region-neutral: an international phone form and
    the ISO IBAN shape, not any one country's domestic formats."""
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="Call me on +33 6 12 34 56 78, pay to FR7630006000011234567890189"),
            Message(role=Role.ASSISTANT, content="Noted"),
        ),
    )

    anonymized, report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    assert "+33 6 12 34 56 78" not in anonymized.messages[0].content
    assert "FR7630006000011234567890189" not in anonymized.messages[0].content
    assert dict(report.replacements)["phone"] == 1
    assert dict(report.replacements)["iban"] == 1


def test_anonymize_does_not_mistake_a_signed_amount_for_a_phone_number() -> None:
    """The phone detector excludes dots as separators so a decimal cannot match it."""
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="Adjust the total by +12.50 and 30.00"),
            Message(role=Role.ASSISTANT, content="Noted"),
        ),
    )

    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    assert anonymized.messages[0].content == "Adjust the total by +12.50 and 30.00"


def test_anonymize_same_value_gets_same_fake_in_text_and_arguments() -> None:
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="Pay to account 81544756 please"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name="create_payment_draft", arguments={"account_number": "81544756"}),),
            ),
            Message(role=Role.TOOL, content='{"account_number": "81544756", "status": "ok"}'),
            Message(role=Role.ASSISTANT, content="Done"),
        ),
    )

    anonymized, report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    fake = anonymized.messages[1].tool_calls[0].arguments["account_number"]
    assert fake != "81544756"
    assert f"account {fake} please" in anonymized.messages[0].content
    assert f'"account_number": "{fake}"' in anonymized.messages[2].content
    assert dict(report.replacements)["account_number"] == 3


def test_anonymize_replaces_integer_financial_arguments() -> None:
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="Pay"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name="create_payment_draft", arguments={"account_number": 81544756, "count": 2}),),
            ),
            Message(role=Role.TOOL, content='{"status": "ok"}'),
            Message(role=Role.ASSISTANT, content="Done"),
        ),
    )

    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    arguments = anonymized.messages[1].tool_calls[0].arguments
    assert arguments["account_number"] != 81544756
    assert isinstance(arguments["account_number"], int)
    assert arguments["count"] == 2


def test_anonymize_email_containing_harvested_name_is_replaced_whole() -> None:
    system = "You are a payment assistant.\n- **Last Name**: Smith\nContact: john.smith@acme.co.uk"
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello"),
        ),
    )

    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    scrubbed = anonymized.messages[0].content.lower()
    assert "smith" not in scrubbed
    assert "john" not in scrubbed
    assert "@example.com" in scrubbed


def test_anonymize_leaves_amounts_untouched() -> None:
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="Pay the invoice, total 12345678.90 USD"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name="parse_file_invoice", arguments={"file_id": "f1"}),),
            ),
            Message(
                role=Role.TOOL,
                content='{"amount": "12345678.90", "amount_pennies": "70200862", "account_number": "81544756"}',
            ),
            Message(role=Role.ASSISTANT, content="Total is 12345678.90 USD"),
        ),
    )

    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    assert "12345678.90" in anonymized.messages[0].content
    assert '"amount": "12345678.90"' in anonymized.messages[2].content
    assert '"amount_pennies": "70200862"' in anonymized.messages[2].content
    assert '"account_number": "81544756"' not in anonymized.messages[2].content
    assert "12345678.90" in anonymized.messages[3].content


def test_anonymize_preserves_ids_consistency_across_messages() -> None:
    dialogue = Dialogue(
        dialogue_id="tr-1",
        messages=(
            Message(role=Role.USER, content="Pay John"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name="get_payees", arguments={"query": "John"}),),
            ),
            Message(role=Role.TOOL, content='{"payee_id": "64a1b2c3d4e5f6a7b8c9d0e1"}'),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name="create_payment", arguments={"payee_id": "64a1b2c3d4e5f6a7b8c9d0e1"}),),
            ),
            Message(role=Role.TOOL, content='{"status": "created"}'),
            Message(role=Role.ASSISTANT, content="Done"),
        ),
    )

    anonymized, _report = Anonymizer(salt="v1").anonymize_dialogue(dialogue)

    fake_id = anonymized.messages[3].tool_calls[0].arguments["payee_id"]
    assert fake_id != "64a1b2c3d4e5f6a7b8c9d0e1"
    assert fake_id in anonymized.messages[2].content
