"""Pydantic v2 schemas for SOP workflow state, fixtures, API contracts, and LLM structured outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"


PHASES_ORDER: list[Phase] = [
    Phase.VERIFY_ID,
    Phase.RESOLVE_INTENT,
    Phase.PROCESS_CASE,
    Phase.POST_PROCESS,
]


class SessionState(str, Enum):
    ACTIVE = "active"
    HANDOFF = "handoff"
    CLOSED = "closed"


class ConsentState(str, Enum):
    UNOFFERED = "unoffered"
    OFFERED = "offered"
    SENT = "sent"
    SKIPPED = "skipped"


class Scope(str, Enum):
    IN = "in"
    OUT = "out"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    SAD = "sad"
    CONFUSED = "confused"
    REFUSING = "refusing"


class Topic(str, Enum):
    OVERVIEW = "overview"
    DENIAL = "denial"
    PAYMENT = "payment"
    APPEAL_DEADLINE = "appeal_deadline"
    DOCUMENTS = "documents"
    ALTERNATIVES = "alternatives"
    SUBMISSION_METHOD = "submission_method"
    FILE_FORMAT = "file_format"
    SUBMISSION_TIMING = "submission_timing"
    PROCESSING_TIME = "processing_time"
    RECEIPT = "receipt"
    UNSUPPORTED = "unsupported"


class IdentityField(str, Enum):
    NAME = "name"
    DOB = "dob"
    PHONE = "phone"
    EMAIL = "email"
    LAST4 = "last4"


IDENTITY_LABELS: dict[IdentityField, str] = {
    IdentityField.NAME: "full name",
    IdentityField.DOB: "date of birth",
    IdentityField.PHONE: "phone number",
    IdentityField.EMAIL: "email address",
    IdentityField.LAST4: "last four digits of your SSN or national ID",
}


# ---------------------------------------------------------------------------
# Fixture Data Models
# ---------------------------------------------------------------------------


class Policyholder(BaseModel):
    party_id: str
    name: str
    name_aliases: list[str] = Field(default_factory=list)
    policy_number: str
    dob: str
    id_type: str
    id_last4: str
    phone: str
    phone_aliases: list[str] = Field(default_factory=list)
    email: str
    email_aliases: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    case_id: str
    party_id: str
    case_type: Literal["healthcare", "dental", "auto"]
    created_at: str
    status: Literal["denied", "closed", "open"]
    summary: str
    denial_reason: Optional[str] = None
    documents_needed: list[str] = Field(default_factory=list)
    appeal_deadline: Optional[str] = None
    expected_reimbursement_amount: Optional[str] = None
    allowed_max_amount: Optional[str] = None
    net_pay: Optional[str] = None
    net_fee: Optional[str] = None


class Representative(BaseModel):
    rep_name: str
    relationship: str
    buyer_name: str
    buyer_party_id: str


class FixtureData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    policyholders: list[Policyholder]
    claims: list[Claim]
    guidance: dict[str, Any]
    representatives: list[Representative] = Field(default_factory=list)
    claim_schema: dict[str, Any] = Field(default_factory=dict)
    consent_scenarios: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM Structured Output Schemas (NLU & Grounded Generation)
# ---------------------------------------------------------------------------


class CaseHints(BaseModel):
    case_id: Optional[str] = Field(
        default=None, description="Normalized claim ID such as CL-2048 if mentioned."
    )
    month: Optional[int] = Field(
        default=None, ge=1, le=12, description="Month number (1-12) of the claim if mentioned."
    )
    year: Optional[str] = Field(
        default=None, description="Four-digit year such as 2025 or 2026 if mentioned."
    )
    case_type: Optional[Literal["healthcare", "dental", "auto"]] = Field(
        default=None, description="Normalized insurance claim type if mentioned."
    )
    status: Optional[Literal["denied", "closed", "open"]] = Field(
        default=None, description="Normalized claim status if mentioned."
    )

    def has_any(self) -> bool:
        return any(
            v is not None
            for v in (self.case_id, self.month, self.year, self.case_type, self.status)
        )

    def to_compact_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class TurnAnalysis(BaseModel):
    """Structured NLU classification produced by the LLM (with local fallback)."""

    scope: Scope = Field(
        default=Scope.UNKNOWN,
        description="Whether the message is in-scope for insurance claims support (in, out, mixed, unknown).",
    )
    topic: Optional[Topic] = Field(
        default=None,
        description="Bounded claim support topic addressed by the caller.",
    )
    emotion: Emotion = Field(
        default=Emotion.NEUTRAL,
        description="Primary emotional state expressed by the caller.",
    )
    emotion_context: Optional[str] = Field(
        default=None,
        description="Brief, non-PII summary of why the caller feels this emotion (e.g. 'repeated identity questions', 'upset about claim denial', 'nervous about email privacy').",
    )
    hints: CaseHints = Field(
        default_factory=CaseHints,
        description="Claim selection clues mentioned anywhere in the caller's turn.",
    )
    declined_fields: list[IdentityField] = Field(
        default_factory=list,
        description="Specific identity fields the caller refused or preferred not to share in this turn.",
    )
    refuses_all_verification: bool = Field(
        default=False,
        description="True if the caller explicitly refuses identity verification altogether.",
    )
    asking_for_alternate_field: bool = Field(
        default=False,
        description="True if the caller is asking whether they can provide a different identity field.",
    )
    asking_why_verify: bool = Field(
        default=False,
        description="True if the caller is asking why identity verification or a specific field is required.",
    )
    is_proxy_caller: bool = Field(
        default=False,
        description="True if the caller states they are calling on behalf of someone else (representative, spouse, child, parent).",
    )
    human_requested: bool = Field(
        default=False,
        description="True if the caller explicitly asks to speak with or transfer to a human representative.",
    )
    handoff_response: Optional[Literal["accept", "decline"]] = Field(
        default=None,
        description="If a handoff offer is currently pending, whether the caller accepts or declines the transfer.",
    )
    consent_decision: Optional[Literal["send", "skip"]] = Field(
        default=None,
        description="In POST_PROCESS when email summary consent is offered, whether the caller explicitly chooses to send or skip the email.",
    )
    asking_about_consent: bool = Field(
        default=False,
        description="True if the caller is asking a question or expressing concern about the email summary or whether consent is required.",
    )
    asking_about_email_recipient: bool = Field(
        default=False,
        description="True if the caller is asking which email address the summary goes to or requesting a different email address.",
    )
    switching_claim: bool = Field(
        default=False,
        description="True if the caller is trying to switch to a different claim during POST_PROCESS.",
    )
    expresses_disappointment_only: bool = Field(
        default=False,
        description="True if the caller is only expressing sadness or disappointment about a denial without asking a specific question.",
    )
    generated_empathy: Optional[str] = Field(
        default=None,
        description="Natural, empathetic de-escalation sentence tailored to the caller's emotional state without leaking any claim data or skipping SOP steps.",
    )


class GroundedReplySynthesis(BaseModel):
    """Structured output for LLM-synthesized conversational replies bounded by SOP phase rules."""

    empathy_opening: Optional[str] = Field(
        default=None,
        description="Empathetic acknowledgment and de-escalation when the caller is frustrated, angry, anxious, sad, confused, or refusing.",
    )
    reply_body: str = Field(
        description="Natural conversational response strictly containing only the authorized facts and SOP prompts for the current phase."
    )


# ---------------------------------------------------------------------------
# Session & Wire Models
# ---------------------------------------------------------------------------


class EmailMessage(BaseModel):
    to: str
    subject: str
    body: str


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    phase: Phase
    emotion: Emotion = Emotion.NEUTRAL


class SessionData(BaseModel):
    phase: Phase = Phase.VERIFY_ID
    state: SessionState = SessionState.ACTIVE
    trace: list[Phase] = Field(default_factory=lambda: [Phase.VERIFY_ID])
    pending_field: Optional[IdentityField] = IdentityField.NAME
    identity: dict[str, str] = Field(default_factory=dict)
    avoided: list[IdentityField] = Field(default_factory=list)
    failures: int = 0
    refusals: int = 0
    out_of_scope: int = 0
    handoff_offer: Optional[Literal["verification", "off-topic"]] = None
    party_id: Optional[str] = None
    claim_id: Optional[str] = None
    topic: Optional[Topic] = None
    hints: CaseHints = Field(default_factory=CaseHints)
    discussed: list[str] = Field(default_factory=list)
    consent: ConsentState = ConsentState.UNOFFERED
    summary_needs_explicit_choice: bool = False
    mail: Optional[EmailMessage] = None
    history: list[ConversationTurn] = Field(default_factory=list)
    empathy_counts: dict[str, int] = Field(default_factory=dict)
    touched: float = 0.0
    busy: bool = False
    custom_ai_token: Optional[str] = None
    custom_ai_url: Optional[str] = None
    custom_ai_model: Optional[str] = None


class TurnResponse(BaseModel):
    reply: str
    phase: Phase
    trace: list[Phase]
    state: SessionState
    consent: ConsentState


class SessionOpenResponse(TurnResponse):
    modelEnabled: bool


class ChatRequest(BaseModel):
    message: str


class AIConfigRequest(BaseModel):
    token: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None


class InboxResponse(BaseModel):
    message: EmailMessage
