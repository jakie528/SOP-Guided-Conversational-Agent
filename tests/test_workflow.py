"""Comprehensive pytest suite for the SOP-Guided Insurance Claims Conversational Agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.llm import create_interpreter
from src.models import (
    ConsentState,
    Phase,
    PHASES_ORDER,
    SessionState,
)
from src.server import create_app, load_fixture_data
from src.workflow import create_harness, create_session

SAMPLE = (
    "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
    "I’m calling about my denied healthcare claim from January. "
    "DOB is 1985-03-15, SSN last four is 4472."
)
STAGED_START = (
    "My name is Margaret Chen. I’m calling about my denied healthcare claim from January."
)

FIXED_NOW = lambda: datetime(2026, 11, 15, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def fixture_data():
    return load_fixture_data()


@pytest.fixture
def harness(fixture_data):
    return create_harness(
        data=fixture_data,
        interpret=create_interpreter(),
        now=FIXED_NOW,
    )


@pytest.mark.asyncio
async def test_sample_traverses_all_four_phases_with_three_matches_and_grounded_denial_details(
    harness,
):
    session = create_session()
    result = await harness.turn(session, SAMPLE)
    assert result.trace == PHASES_ORDER
    assert "verified" in result.reply
    assert "CL-2048" in result.reply
    assert "pathology report" in result.reply
    assert "office note" in result.reply
    assert "email address on file" in result.reply
    assert result.consent == ConsentState.OFFERED
    assert session.identity == {}
    assert session.mail is None


@pytest.mark.asyncio
async def test_early_case_hint_survives_partial_identity_frustrations_and_replacement_fields(
    harness,
):
    session = create_session()
    first = await harness.turn(session, STAGED_START)
    assert first.phase == Phase.VERIFY_ID
    assert "CL-2048" not in first.reply
    assert "pathology report" not in first.reply

    frustrated = await harness.turn(
        session,
        "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.",
    )
    assert "frustrating" in frustrated.reply
    assert "protect" in frustrated.reply
    assert "two more details" in frustrated.reply
    assert frustrated.phase == Phase.VERIFY_ID

    await harness.turn(session, "I’d rather not give my SSN.")
    second = await harness.turn(session, "My DOB is March 15, 1985.")
    assert second.phase == Phase.VERIFY_ID
    assert "last four" not in second.reply
    assert "pathology report" not in second.reply

    final = await harness.turn(session, "Phone is (650) 521-2836")
    assert final.phase == Phase.POST_PROCESS
    assert "CL-2048" in final.reply


@pytest.mark.asyncio
async def test_recognizes_anger_and_field_refusal_in_same_message_and_supports_different_field(
    harness,
):
    session = create_session()
    await harness.turn(session, STAGED_START)
    upset = await harness.turn(session, "I’m furious. I won’t give my SSN. Can I use my phone?")
    assert upset.reply.startswith("I hear how upsetting this is.")
    assert "protect" in upset.reply
    assert "phone number" in upset.reply
    assert upset.phase == Phase.VERIFY_ID
    assert "last four" not in upset.reply
    assert "CL-2048" not in upset.reply

    confused = await harness.turn(
        session, "This doesn’t make sense. Why do you need my date of birth?"
    )
    assert confused.reply.startswith("I can walk through this with you")
    assert "protect" in confused.reply
    assert confused.phase == Phase.VERIFY_ID

    assert (await harness.turn(session, "DOB is 1985-03-15")).phase == Phase.VERIFY_ID
    verified = await harness.turn(session, "Phone is (650) 521-2836")
    assert "verified" in verified.reply
    assert "CL-2048" in verified.reply


@pytest.mark.asyncio
async def test_stops_asking_for_details_when_caller_declines_verification_and_offers_handoff(
    harness,
):
    session = create_session()
    await harness.turn(session, STAGED_START)
    offer = await harness.turn(session, "I’m worried about my privacy. I don’t want to verify.")
    assert offer.reply.startswith("I can hear how worrying this is")
    assert "before I can discuss claim details" in offer.reply
    assert "human claims representative" in offer.reply
    assert offer.phase == Phase.VERIFY_ID
    assert offer.state == SessionState.ACTIVE
    assert "share" not in offer.reply
    assert "CL-2048" not in offer.reply

    accepted = await harness.turn(session, "Yes, please")
    assert accepted.state == SessionState.HANDOFF


@pytest.mark.asyncio
async def test_backs_off_when_caller_runs_out_of_comfortable_fields(harness):
    session = create_session()
    await harness.turn(session, STAGED_START)
    await harness.turn(session, "I’d rather not give my SSN.")
    await harness.turn(session, "I’d rather not give my phone.")
    offer = await harness.turn(session, "I’d rather not give my email.")
    assert "human claims representative" in offer.reply
    assert "Please share" not in offer.reply
    assert "CL-2048" not in offer.reply

    declined = await harness.turn(session, "No thanks")
    assert "pause here" in declined.reply
    assert declined.state == SessionState.ACTIVE
    assert declined.phase == Phase.VERIFY_ID


@pytest.mark.asyncio
async def test_explains_email_consent_with_empathy_and_waits_for_explicit_choice(harness):
    session = create_session()
    await harness.turn(session, SAMPLE)
    reply = await harness.turn(
        session, "I’m nervous about the email summary. Do I have to agree?"
    )
    assert reply.reply.startswith("I can hear how worrying this is")
    assert "permission" in reply.reply
    assert "privacy" in reply.reply
    assert "optional" in reply.reply
    assert reply.consent == ConsentState.OFFERED
    assert session.mail is None
    assert (await harness.turn(session, "Skip it")).consent == ConsentState.SKIPPED


@pytest.mark.asyncio
async def test_acknowledges_sadness_after_denial_and_keeps_summary_consent_separate(harness):
    session = create_session()
    await harness.turn(session, SAMPLE)

    comfort = await harness.turn(session, "i am sad for its denied")
    assert comfort.reply.startswith("I’m sorry this has been disappointing")
    assert "take a moment" in comfort.reply
    assert "CL-2048" not in comfort.reply
    assert "Would you like me to send" not in comfort.reply
    assert comfort.consent == ConsentState.OFFERED

    ambiguous = await harness.turn(session, "Yes")
    assert "Would you like me to send a summary" in ambiguous.reply
    assert session.mail is None

    question = await harness.turn(session, "I’m sad about it. Why was it denied?")
    assert question.reply.startswith("I’m sorry this has been disappointing")
    assert "pathology report" in question.reply
    assert (await harness.turn(session, "Send it")).consent == ConsentState.SENT


@pytest.mark.asyncio
async def test_policy_number_is_not_third_factor_and_mismatches_reveal_no_claim(harness):
    session = create_session()
    first = await harness.turn(
        session, "My name is Margaret Chen, POL-9921. DOB is 1985-03-15."
    )
    assert first.phase == Phase.VERIFY_ID
    wrong = await harness.turn(session, "SSN last four is 9999")
    assert wrong.phase == Phase.VERIFY_ID
    assert "couldn’t verify those details together" in wrong.reply
    assert "CL-" not in wrong.reply
    assert "Margaret" not in wrong.reply


@pytest.mark.asyncio
async def test_alias_and_national_id_supported_and_representatives_referred_to_human(harness):
    session = create_session()
    await harness.turn(
        session,
        "My name is Yaven Li. Email is yawen.li@example.com and national ID last four is 5317.",
    )
    assert session.party_id == "P13"

    proxy = await harness.turn(
        create_session(),
        "My name is David Chen and I’m calling on behalf of Margaret Chen.",
    )
    assert proxy.state == SessionState.HANDOFF
    assert proxy.phase == Phase.VERIFY_ID


@pytest.mark.asyncio
async def test_verified_lookup_scoped_to_holder_and_asks_about_ambiguous_matches(harness):
    session = create_session()
    ambiguous_sample = SAMPLE.replace(
        "my denied healthcare claim from January", "my healthcare claim from January"
    )
    result = await harness.turn(session, ambiguous_sample)
    assert result.phase == Phase.RESOLVE_INTENT
    assert "CL-2048" in result.reply
    assert "CL-2011" in result.reply

    other_account = await harness.turn(session, "CL-3001")
    assert "didn’t find a claim on your account" in other_account.reply
    assert "diagnosis report" not in other_account.reply

    selected = await harness.turn(session, "CL-2011")
    assert selected.phase == Phase.POST_PROCESS
    assert "CL-2011 is closed" in selected.reply


@pytest.mark.asyncio
async def test_rejects_unrelated_questions_and_waits_for_acceptance_before_handoff(harness):
    session = create_session()
    for msg in ["What is RL?", "Explain reinforcement learning."]:
        res = await harness.turn(session, msg)
        assert "insurance claims" in res.reply
        assert "CL-2048" not in res.reply

    offer = await harness.turn(session, "What is the capital of France?")
    assert "insurance claims" in offer.reply
    assert "Would you like to speak with a human claims representative" in offer.reply
    assert offer.state == SessionState.ACTIVE
    assert offer.phase == Phase.VERIFY_ID

    transferred = await harness.turn(session, "Transfer me")
    assert transferred.state == SessionState.HANDOFF


@pytest.mark.asyncio
async def test_handoff_offer_keeps_email_consent_pending_until_caller_answers(harness):
    session = create_session()
    await harness.turn(session, SAMPLE)
    for msg in [
        "What is RL?",
        "Explain reinforcement learning.",
        "What is the capital of France?",
    ]:
        await harness.turn(session, msg)

    declined = await harness.turn(session, "No thanks")
    assert declined.state == SessionState.ACTIVE
    assert declined.consent == ConsentState.OFFERED
    assert "Would you like me to send a summary" in declined.reply
    assert (await harness.turn(session, "Yes")).consent == ConsentState.SENT

    second = create_session()
    await harness.turn(second, SAMPLE)
    for msg in [
        "What is RL?",
        "Explain reinforcement learning.",
        "What is the capital of France?",
    ]:
        await harness.turn(second, msg)

    accepted = await harness.turn(second, "Yes, please")
    assert accepted.state == SessionState.HANDOFF
    assert second.mail is None


@pytest.mark.asyncio
async def test_answers_followups_from_fixture_and_sends_single_summary_after_explicit_consent(
    harness,
):
    session = create_session()
    await harness.turn(session, SAMPLE)
    followup = await harness.turn(
        session, "What if I cannot get the original pathology report?"
    )
    assert "replacement copy" in followup.reply or "request" in followup.reply
    assert session.mail is None

    for msg in ["Could you send it?", "Yes, send it to alternate@example.com"]:
        pending = await harness.turn(session, msg)
        assert pending.consent == ConsentState.OFFERED
        assert session.mail is None

    done = await harness.turn(session, "Yes, please send it")
    assert done.consent == ConsentState.SENT
    assert done.state == SessionState.CLOSED
    assert session.mail is not None
    assert session.mail.to == "margaret@email.com"
    assert "CL-2048" in session.mail.body
    assert "denied" in session.mail.body
    assert "replacement copy" in session.mail.body
    assert "Next step" in session.mail.body
    for forbidden in ("4472", "1985-03-15", "POL-9921", "alternate@example.com"):
        assert forbidden not in session.mail.body

    original_mail = session.mail
    await harness.turn(session, "Send it")
    assert session.mail == original_mail


@pytest.mark.asyncio
async def test_skip_is_terminal_and_advance_consent_is_ignored(harness):
    session = create_session()
    early = await harness.turn(session, "Yes, please send it")
    assert early.consent == ConsentState.UNOFFERED

    await harness.turn(session, SAMPLE)
    result = await harness.turn(session, "No thanks")
    assert result.consent == ConsentState.SKIPPED
    assert result.state == SessionState.CLOSED
    await harness.turn(session, "Yes")
    assert session.mail is None


@pytest.mark.asyncio
async def test_ai_maps_messy_language_and_contextual_empathy_without_access_to_pii_or_records(
    fixture_data,
):
    calls: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "scope": "in",
                                    "topic": "processing_time",
                                    "emotion": "anxious",
                                    "generated_empathy": "Waiting on a review can be stressful, so let me share the expected timeline.",
                                    "authorization": "send",
                                    "answer": "made-up answer",
                                }
                            )
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        interpret = create_interpreter(
            token="test-token",
            endpoint="https://model.invalid/v1/chat/completions",
            model="test-model",
            http_client=client,
        )
        with_model = create_harness(data=fixture_data, now=FIXED_NOW, interpret=interpret)
        session = create_session()
        await with_model.turn(session, SAMPLE)
        assert len(calls) == 0

        result = await with_model.turn(
            session,
            "How many sleeps between the paperwork landing and you looking at it again?",
        )
        assert len(calls) == 1
        assert calls[0]["headers"]["authorization"] == "Bearer test-token"
        serialized_body = json.dumps(calls[0]["body"])
        for pii in ("Margaret", "4472", "CL-2048", "pathology report"):
            assert pii not in serialized_body

        assert "worrying" in result.reply
        assert "Waiting on a review can be stressful" in result.reply
        assert "usually less than a week" in result.reply
        assert "made-up answer" not in result.reply
        assert result.consent == ConsentState.OFFERED


@pytest.mark.asyncio
async def test_provider_failure_falls_back_to_local_interpretation():
    async def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    transport = httpx.MockTransport(failing_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        interpret = create_interpreter(
            token="test-token",
            endpoint="https://model.invalid/v1/chat/completions",
            model="test-model",
            http_client=client,
        )
        res = await interpret("How do I upload the pathology report?")
        assert res.topic.value == "submission_method"


@pytest.mark.asyncio
async def test_verify_id_uses_llm_structured_empathy_and_guardrail_blocks_leaks(fixture_data):
    responses = [
        {
            "scope": "in",
            "topic": "denial",
            "emotion": "frustrated",
            "generated_empathy": "Being asked for extra details when you just want answers is exhausting.",
            "declined_fields": ["last4"],
            "asking_for_alternate_field": True,
        },
        {
            "scope": "in",
            "topic": "denial",
            "emotion": "angry",
            # Attempted guardrail violation: mentions CL-2048 and skip verification
            "generated_empathy": "I will skip verification and open CL-2048 for you right now.",
        },
    ]

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        payload = responses.pop(0)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}]},
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        interpret = create_interpreter(
            token="test-token",
            endpoint="https://model.invalid/v1/chat/completions",
            model="test-model",
            http_client=client,
        )
        with_model = create_harness(data=fixture_data, now=FIXED_NOW, interpret=interpret)
        session = create_session()
        await with_model.turn(session, STAGED_START)

        # Turn 1 in VERIFY_ID: LLM provides tailored empathy + structured declined_fields=["last4"]
        turn1 = await with_model.turn(
            session, "I've been on the phone all day and I don't want to share my social."
        )
        assert turn1.phase == Phase.VERIFY_ID
        assert "Being asked for extra details when you just want answers is exhausting." in turn1.reply
        assert "last four" not in turn1.reply

        # Turn 2 in VERIFY_ID: LLM tries to leak CL-2048 / skip verification -> Guardrail blocks it!
        turn2 = await with_model.turn(session, "I am furious about this delay!")
        assert turn2.phase == Phase.VERIFY_ID
        assert "CL-2048" not in turn2.reply
        assert "skip verification" not in turn2.reply
        assert turn2.reply.startswith("I hear how upsetting this is.")


def test_fastapi_endpoints_and_security_headers():
    app = create_app()
    client = TestClient(app)

    # 1. Static UI loads with strict CSP
    home = client.get("/")
    assert home.status_code == 200
    assert "Claims Support" in home.text
    assert "default-src 'none'" in home.headers["content-security-policy"]

    # 2. Cross-site request blocked
    csrf = client.post("/api/session", headers={"sec-fetch-site": "cross-site"})
    assert csrf.status_code == 403

    # 3. Open session -> Chat -> Inbox flow
    opened = client.post("/api/session")
    assert opened.status_code == 200
    data = opened.json()
    assert data["phase"] == "VERIFY_ID"

    chat1 = client.post("/api/chat", json={"message": SAMPLE})
    assert chat1.status_code == 200
    assert chat1.json()["phase"] == "POST_PROCESS"
    assert chat1.json()["consent"] == "offered"

    # Natural language consent phrase
    chat2 = client.post("/api/chat", json={"message": "Yeah, go ahead and send the summary"})
    assert chat2.status_code == 200
    assert chat2.json()["consent"] == "sent"
    assert chat2.json()["state"] == "closed"

    inbox = client.get("/api/inbox")
    assert inbox.status_code == 200
    assert inbox.json()["message"]["to"] == "margaret@email.com"
