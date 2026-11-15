"""SOP-Guided Conversational Agent state machine and phase-bounded orchestration."""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .claims import claim_answer, describe_case, format_list, match_cases, summary_for
from .identity import (
    FIELD_MENTION_PATTERNS,
    extract_identity,
    is_proxy_or_representative,
    redact_pii,
    verify_identity,
)
from .llm import (
     InterpreterFn,
    analyze_turn_locally,
    compose_empathy,
    extract_refusal_details,
    human_requested,
    parse_consent_choice,
)
from .models import (
    IDENTITY_LABELS,
    PHASES_ORDER,
    CaseHints,
    ConsentState,
    ConversationTurn,
    EmailMessage,
    Emotion,
    FixtureData,
    IdentityField,
    Phase,
    Scope,
    SessionData,
    SessionState,
    Topic,
    TurnAnalysis,
    TurnResponse,
)

GREETING = (
    "Hi, I can help you with an insurance claim. To protect your information, "
    "I’ll need to match three identity details before opening a claim. What’s your full name?"
)
CONSENT_PROMPT = (
    "Would you like me to send a summary to the email address on file? "
    "You can say “send it” or “skip it.”"
)
VERIFICATION_HANDOFF_PROMPT = (
    "I need to verify your identity before I can discuss claim details. "
    "Would you like to speak with a human claims representative? You can say “transfer me.”"
)
CONSENT_EXPLANATION = (
    "The summary includes claim details. I ask for your permission before sending it "
    "to protect your privacy. The email is optional."
)
SCOPE_MESSAGE = "I can only help with insurance claims and related support questions."


def create_session() -> SessionData:
    """Create a fresh SOP session initialized at VERIFY_ID."""
    return SessionData()


class SOPHarness:
    """Deterministic SOP harness with LLM-native structured NLU and grounded emotional support."""

    def __init__(
        self,
        data: FixtureData | dict[str, Any],
        interpret: InterpreterFn,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.data = data if isinstance(data, FixtureData) else FixtureData.model_validate(data)
        self.interpret = interpret
        self.now = now
        self._interpret_arity = len(inspect.signature(interpret).parameters)

    def open(self) -> TurnResponse:
        session = create_session()
        return self._response(session, GREETING)

    async def _call_interpreter(
        self, safe_text: str, raw_message: str, session: SessionData
    ) -> TurnAnalysis:
        if self._interpret_arity >= 3:
            res = await self.interpret(safe_text, raw_message, session)  # type: ignore[call-arg]
        else:
            res = await self.interpret(safe_text)
        if isinstance(res, TurnAnalysis):
            return res
        if isinstance(res, dict):
            return TurnAnalysis.model_validate(res)
        return analyze_turn_locally(safe_text, raw_message=raw_message, session=session)

    @staticmethod
    def _response(session: SessionData, reply: str) -> TurnResponse:
        return TurnResponse(
            reply=reply,
            phase=session.phase,
            trace=list(session.trace),
            state=session.state,
            consent=session.consent,
        )

    @staticmethod
    def _advance(session: SessionData, target_phase: Phase) -> None:
        current_idx = PHASES_ORDER.index(session.phase)
        target_idx = PHASES_ORDER.index(target_phase)
        if target_idx != current_idx + 1:
            raise ValueError(f"Invalid phase transition: {session.phase} -> {target_phase}")
        if not session.party_id:
            raise ValueError("Identity verification required before advancing phase")
        if target_phase != Phase.RESOLVE_INTENT and not session.claim_id:
            raise ValueError("Claim selection required before advancing to case processing")
        session.phase = target_phase
        session.trace.append(target_phase)

    def _handoff(self, session: SessionData, reason: str) -> TurnResponse:
        session.state = SessionState.HANDOFF
        session.identity.clear()
        if session.consent == ConsentState.OFFERED:
            session.consent = ConsentState.SKIPPED
        return self._response(
            session,
            f"{reason} I’ve requested a handoff to a human claims representative in this demo.",
        )

    @staticmethod
    def _remember(session: SessionData, analysis: TurnAnalysis) -> None:
        compact = analysis.hints.to_compact_dict()
        if compact.get("case_id"):
            session.hints = CaseHints(case_id=compact["case_id"])
        elif compact:
            if session.hints.case_id:
                session.hints = CaseHints()
            merged = session.hints.to_compact_dict()
            merged.update(compact)
            session.hints = CaseHints.model_validate(merged)

        selecting_case = bool(compact) and analysis.topic == Topic.OVERVIEW
        if analysis.topic and (not selecting_case or not session.topic):
            session.topic = analysis.topic

    def _handle_verify_id(
        self,
        session: SessionData,
        message: str,
        fields: dict[str, str],
        analysis: TurnAnalysis,
    ) -> Optional[TurnResponse]:
        lead = compose_empathy(analysis, session=session, phase=Phase.VERIFY_ID)

        if is_proxy_or_representative(message, self.data.representatives):
            return self._handoff(
                session,
                "A representative can verify your identity and authority to act for the policyholder.",
            )

        refuses_detail, refuses_all, declined_fields, asking_alternate = extract_refusal_details(
            message, session.pending_field
        )
        if analysis.refuses_all_verification:
            refuses_all = True
            refuses_detail = True
        if analysis.declined_fields:
            for f in analysis.declined_fields:
                if f not in declined_fields:
                    declined_fields.append(f)

        if refuses_detail:
            fields_to_avoid = (
                declined_fields
                if declined_fields
                else ([session.pending_field] if session.pending_field else [])
            )
            for field in fields_to_avoid:
                if field and field not in session.avoided:
                    session.avoided.append(field)

        valid_label_keys = {f.value for f in IdentityField}
        made_progress = any(
            k in valid_label_keys and k not in session.identity for k in fields.keys()
        )
        if made_progress:
            session.refusals = 0
        elif refuses_detail:
            session.refusals += 1

        if refuses_all:
            session.handoff_offer = "verification"
            return self._response(
                session,
                " ".join([lead or "I respect your choice.", VERIFICATION_HANDOFF_PROMPT]),
            )

        session.identity.update(fields)
        holder = verify_identity(session.identity, self.data.policyholders)
        if holder:
            session.party_id = holder.party_id
            session.identity.clear()
            self._advance(session, Phase.RESOLVE_INTENT)
            return None

        given = [
            IdentityField(k) for k in session.identity.keys() if k in valid_label_keys
        ]
        if len(given) >= 3:
            session.failures += 1
            session.identity.clear()
            session.pending_field = IdentityField.NAME
            if session.failures >= 3:
                return self._handoff(session, "I wasn’t able to complete verification here.")
            parts = [
                lead,
                "I couldn’t verify those details together. To protect your claim information, "
                "please try three details as they appear on the policy. You can use your full name, "
                "date of birth, phone, email, or the last four digits of your SSN or national ID. "
                "A representative can help if you’d prefer.",
            ]
            return self._response(session, " ".join(p for p in parts if p))

        available = [
            f for f in IdentityField if f not in given and f not in session.avoided
        ]
        if session.refusals >= 3 or len(available) + len(given) < 3:
            session.handoff_offer = "verification"
            return self._response(
                session,
                " ".join([lead or "I understand.", VERIFICATION_HANDOFF_PROMPT]),
            )

        clarifying = (
            asking_alternate
            or analysis.asking_for_alternate_field
            or analysis.asking_why_verify
        )
        protection = (
            "I need to match three identity details before I can discuss your claim. "
            "That helps protect your claim information."
        )
        if given or lead or clarifying:
            session.pending_field = available[0]
            choices = format_list([IDENTITY_LABELS[f] for f in available], "or")
            count = 3 - len(given)
            count_phrase = (
                "one more detail"
                if count == 1
                else "two more details"
                if count == 2
                else "three details"
            )
            prompt = (
                f"Please share {count_phrase} using your {choices}. "
                "I can also connect you with a representative."
            )
        else:
            session.pending_field = IdentityField.NAME
            prompt = (
                "What’s your full name? You can also start with your date of birth, phone, "
                "email, or the last four digits of your SSN or national ID."
            )

        return self._response(
            session, " ".join(p for p in (lead, protection, prompt) if p)
        )

    def _handle_resolve_intent(self, session: SessionData) -> Optional[TurnResponse]:
        assert session.party_id is not None
        candidates = match_cases(self.data.claims, session.party_id, session.hints)
        if len(candidates) == 1:
            session.claim_id = candidates[0].case_id
            self._advance(session, Phase.PROCESS_CASE)
            return None

        if len(candidates) == 0:
            owned = match_cases(self.data.claims, session.party_id, CaseHints())
            session.hints = CaseHints()
            if len(owned) == 0:
                return self._response(
                    session,
                    "I don’t see a claim on your verified account. "
                    "I can connect you with a claims representative for help.",
                )
            return self._response(
                session,
                "I didn’t find a claim on your account matching those details. "
                "Which claim did you mean? You can give me a claim number, type, or month.",
            )

        descriptions = "\n".join(describe_case(c) for c in candidates)
        return self._response(
            session,
            "Which claim would you like to discuss? You can give me its claim number "
            f"or add a detail about the date or type.\n\n{descriptions}",
        )

    async def turn(self, session: SessionData, message: str) -> TurnResponse:
        """Execute a single conversational turn under SOP phase and safety constraints."""
        if session.state != SessionState.ACTIVE:
            return self._response(
                session,
                "This chat has ended. Start a new chat if you need help with a claim.",
            )

        # 1. Check pending handoff offer response
        if session.handoff_offer:
            if re.match(r"^\s*(?:yes|sure|okay|ok)(?:,? please)?[.!]?\s*$", message, re.I):
                return self._handoff(session, "Of course.")
            if re.match(
                r"^\s*(?:no(?:,? (?:thanks|thank you))?|no transfer|keep chatting|"
                r"please don['’]t transfer me|don['’]t transfer me)[.!]?\s*$",
                message,
                re.I,
            ):
                was_verification = session.handoff_offer == "verification"
                session.handoff_offer = None
                session.out_of_scope = 0
                if was_verification:
                    return self._response(
                        session,
                        "We can pause here. If you decide to continue, share an identity detail "
                        "you’re comfortable using.",
                    )
                follow_up = f" {CONSENT_PROMPT}" if session.consent == ConsentState.OFFERED else ""
                return self._response(
                    session, f"We can keep chatting here about your claim.{follow_up}"
                )

        # 2. Check explicit human transfer request
        if human_requested(message):
            return self._handoff(session, "Of course.")
        session.handoff_offer = None

        # 3. Extract & Redact PII before NLU
        verifying = session.phase == Phase.VERIFY_ID
        fields, matched = extract_identity(
            message, session.pending_field if verifying else None
        )
        safe_text = redact_pii(message, matched)
        local = analyze_turn_locally(safe_text, raw_message=message, session=session)

        # 4. Handle POST_PROCESS explicit email consent decisions
        choice = (
            session.phase == Phase.POST_PROCESS
            and session.consent == ConsentState.OFFERED
            and parse_consent_choice(
                message, allow_short_reply=not session.summary_needs_explicit_choice
            )
        )
        if (
            session.summary_needs_explicit_choice
            and not choice
            and parse_consent_choice(message, allow_short_reply=True)
        ):
            session.summary_needs_explicit_choice = False
            return self._response(session, CONSENT_PROMPT)

        session.summary_needs_explicit_choice = False
        if choice:
            assert session.party_id is not None
            matched_claims = match_cases(
                self.data.claims, session.party_id, CaseHints(case_id=session.claim_id)
            )
            claim = matched_claims[0] if matched_claims else None
            holder = next(
                (h for h in self.data.policyholders if h.party_id == session.party_id), None
            )
            if choice == "send" and claim and holder and holder.email:
                session.mail = EmailMessage(
                    to=holder.email,
                    subject=f"Summary for claim {claim.case_id}",
                    body=summary_for(claim, session.discussed, self.now()),
                )
                session.consent = ConsentState.SENT
            else:
                session.consent = ConsentState.SKIPPED
            session.state = SessionState.CLOSED

            if session.consent == ConsentState.SENT:
                return self._response(
                    session,
                    "I sent the summary to the email address on file. You can see it in the demo inbox.",
                )
            if choice == "send":
                return self._response(
                    session,
                    "I couldn’t deliver the summary to an email on file. "
                    "A claims representative can help you with a copy.",
                )
            return self._response(
                session, "I’ll skip the email. Thanks for contacting claims support."
            )

        # 5. Handle POST_PROCESS questions/anxiety about email consent or disappointment
        if (
            session.consent == ConsentState.OFFERED
            and local.asking_about_consent
            and local.scope not in (Scope.OUT, Scope.MIXED)
        ):
            session.out_of_scope = 0
            lead = compose_empathy(local, session=session, phase=session.phase)
            return self._response(
                session,
                " ".join(p for p in (lead, CONSENT_EXPLANATION, CONSENT_PROMPT) if p),
            )

        if (
            session.phase == Phase.POST_PROCESS
            and local.expresses_disappointment_only
            and local.scope not in (Scope.OUT, Scope.MIXED)
        ):
            session.out_of_scope = 0
            session.summary_needs_explicit_choice = True
            lead = compose_empathy(local, session=session, phase=session.phase)
            return self._response(
                session,
                f"{lead} We can take a moment. Whenever you’re ready, I can help you understand your options.",
            )

        # 6. Invoke LLM Structured Turn Analysis (or local fallback)
        shorthand = bool(
            not verifying
            and session.claim_id
            and re.match(
                r"^\s*(?:why (?:was that|is that)|why|what does that mean|what happened)[?.!\s]*$",
                message,
                re.I,
            )
        )
        # In VERIFY_ID, turns containing extracted PII stay local;
        # non-PII emotional/refusal/clarification turns in VERIFY_ID and post-verification turns use the LLM interpreter.
        needs_llm_in_verify = (
            verifying
            and not fields
            and local.scope not in (Scope.OUT, Scope.MIXED)
            and not re.search(
                r"\?|^\s*(?:what|how|why|when|where|who|which|tell me|explain|define)\b",
                safe_text,
                re.I,
            )
            or (
                verifying
                and not fields
                and (
                    local.emotion != Emotion.NEUTRAL
                    or local.asking_why_verify
                    or local.asking_for_alternate_field
                    or bool(local.declined_fields)
                )
            )
        )
        if shorthand:
            analysis = local.model_copy(update={"scope": Scope.IN, "topic": Topic.OVERVIEW})
        elif verifying and not needs_llm_in_verify:
            analysis = local
        else:
            analysis = await self._call_interpreter(safe_text, message, session)

        # Record redacted turn in history for multi-turn awareness
        session.history.append(
            ConversationTurn(
                role="user",
                text=safe_text,
                phase=session.phase,
                emotion=analysis.emotion,
            )
        )

        # 7. Enforce Scope Guardrails & 3-Strike Handoff Offer
        scope = analysis.scope
        lead = compose_empathy(analysis, session=session, phase=session.phase)
        claim_selection = session.phase == Phase.RESOLVE_INTENT and local.hints.has_any()
        identity_conversation = verifying and (
            bool(fields)
            or bool(lead)
            or bool(
                re.search(
                    r"\b(?:verification|verify|full name|birth|dob|ssn|last four|email|phone|"
                    r"identify|identification|protect|identity|privacy|policyholder)\b",
                    message,
                    re.I,
                )
            )
        )
        conversational = bool(
            re.match(
                r"^(?:hi|hello|hey|thanks|thank you|got it|okay|ok|yes|no|sure|please)[.!\s]*$",
                message.strip(),
                re.I,
            )
        )
        unrelated = (
            scope in (Scope.OUT, Scope.MIXED)
            or (
                scope == Scope.UNKNOWN
                and not identity_conversation
                and not claim_selection
                and not conversational
                and bool(
                    re.search(
                        r"\?|^\s*(?:what|how|why|when|where|who|which|tell me|explain|define)\b",
                        safe_text,
                        re.I,
                    )
                )
            )
        )

        if unrelated:
            session.out_of_scope += 1
            if session.out_of_scope >= 3:
                session.handoff_offer = "off-topic"
                return self._response(
                    session,
                    f"{SCOPE_MESSAGE} Would you like to speak with a human claims representative? "
                    "You can say “transfer me” or keep chatting here about your claim.",
                )
            if scope != Scope.MIXED:
                invitation = (
                    CONSENT_PROMPT
                    if session.consent == ConsentState.OFFERED
                    else "We can continue identity verification whenever you’re ready."
                    if verifying
                    else "What would you like help with on your claim?"
                )
                return self._response(session, f"{SCOPE_MESSAGE} {invitation}")
        else:
            session.out_of_scope = 0

        prefix = " ".join(p for p in (lead, SCOPE_MESSAGE if unrelated else None) if p)

        # 8. Phase 1: VERIFY_ID
        if verifying:
            case_hint = analysis.hints.has_any()
            claim_question = bool(
                re.search(r"\b(?:claim|appeal|reimbursement|denied)\b", safe_text, re.I)
            )
            if scope != Scope.OUT and (case_hint or claim_question):
                self._remember(session, analysis)

            verify_result = self._handle_verify_id(session, message, fields, analysis)
            if verify_result is not None:
                if unrelated:
                    verify_result.reply = f"{SCOPE_MESSAGE} {verify_result.reply}"
                return verify_result

            prefix = " ".join(p for p in (prefix, "Thanks, your identity is verified.") if p)
            if not session.topic and not session.hints.has_any():
                return self._response(
                    session, f"{prefix} What would you like help with on your claim?"
                )

        # 9. Phase 2: RESOLVE_INTENT
        if session.phase == Phase.RESOLVE_INTENT:
            if not verifying:
                if scope not in (Scope.IN, Scope.MIXED) and not claim_selection:
                    return self._response(
                        session,
                        " ".join(
                            p
                            for p in (
                                prefix,
                                "Tell me what you need help with or give me the claim number, type, or month.",
                            )
                            if p
                        ),
                    )
                self._remember(session, analysis)

            resolve_result = self._handle_resolve_intent(session)
            if resolve_result is not None:
                resolve_result.reply = " ".join(
                    p for p in (prefix, resolve_result.reply) if p
                )
                return resolve_result

        # 10. Phase 3 & 4: PROCESS_CASE & POST_PROCESS
        assert session.party_id is not None
        matched_claims = match_cases(
            self.data.claims, session.party_id, CaseHints(case_id=session.claim_id)
        )
        claim = matched_claims[0] if matched_claims else None
        if not claim:
            return self._handoff(session, "I couldn’t open the selected claim.")

        if not verifying and session.phase == Phase.POST_PROCESS and local.switching_claim:
            return self._response(
                session,
                "We can finish the summary for this claim first, then you can start a new chat "
                f"for another claim. {CONSENT_PROMPT}",
            )

        if (
            not verifying
            and session.phase == Phase.POST_PROCESS
            and local.hints.has_any()
            and not match_cases([claim], session.party_id, local.hints)
            and (
                local.hints.case_id
                or local.hints.case_type
                or (local.hints.month and re.search(r"\bclaim\b", message, re.I))
            )
        ):
            return self._response(
                session,
                f"This chat is about {claim.case_id}. To discuss another claim, start a new chat "
                f"or ask for a representative. {CONSENT_PROMPT}",
            )

        if (
            session.phase == Phase.POST_PROCESS
            and (
                conversational
                or re.search(r"\b(?:email|summary|send it|skip it)\b", safe_text, re.I)
            )
            and (
                scope != Scope.IN
                or not analysis.topic
                or analysis.topic == Topic.UNSUPPORTED
            )
        ):
            return self._response(
                session,
                " ".join(
                    p
                    for p in (
                        prefix,
                        "The summary covers your claim status, what we discussed, and next steps. "
                        "I can send it to the address on file with your permission.",
                        CONSENT_PROMPT,
                    )
                    if p
                ),
            )

        topic = (
            (session.topic or Topic.OVERVIEW)
            if session.phase == Phase.PROCESS_CASE
            else (analysis.topic or Topic.UNSUPPORTED)
        )

        if session.phase == Phase.POST_PROCESS and local.asking_about_email_recipient:
            return self._response(
                session,
                "I can send the summary to the email already on file. "
                f"A representative can help with address changes. {CONSENT_PROMPT}",
            )

        answer = claim_answer(claim, topic, safe_text, self.data.guidance, self.now())
        if topic != Topic.UNSUPPORTED and len(session.discussed) < 20:
            session.discussed.append(answer)

        if session.phase == Phase.PROCESS_CASE:
            self._advance(session, Phase.POST_PROCESS)
            session.consent = ConsentState.OFFERED

        final_reply = "\n\n".join(p for p in (prefix, answer, CONSENT_PROMPT) if p)
        session.history.append(
            ConversationTurn(
                role="assistant",
                text=final_reply,
                phase=session.phase,
                emotion=Emotion.NEUTRAL,
            )
        )
        return self._response(session, final_reply)


def create_harness(
    data: FixtureData | dict[str, Any],
    interpret: InterpreterFn,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> SOPHarness:
    """Factory function returning an initialized SOPHarness instance."""
    return SOPHarness(data=data, interpret=interpret, now=now)
