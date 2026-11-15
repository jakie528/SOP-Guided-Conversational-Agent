"""Deterministic claim filtering, grounded answer synthesis, and email summary formatting."""

from __future__ import annotations

import calendar
from datetime import datetime
from typing import Any

from .models import CaseHints, Claim, Topic

DOCUMENT_KEYS: dict[str, str] = {
    "pathology report": "original pathology report",
    "office note": "treating provider office note",
}

GUIDANCE_TOPICS: dict[Topic, str] = {
    Topic.SUBMISSION_METHOD: "submission_method",
    Topic.FILE_FORMAT: "file_format_requirements",
    Topic.SUBMISSION_TIMING: "submission_timing",
    Topic.PROCESSING_TIME: "processing_time_after_submission",
    Topic.RECEIPT: "receipt_confirmation",
}


def format_list(items: list[str], conjunction: str = "and") -> str:
    """Format a list of strings using standard English Oxford-comma conjunction/disjunction."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


def format_currency(amount_str: str) -> str:
    value = float(amount_str)
    return f"${value:,.2f}"


def match_cases(claims: list[Claim], party_id: str, hints: CaseHints) -> list[Claim]:
    """Filter claims strictly belonging to the verified party_id and matching any case hints."""
    matched: list[Claim] = []
    for claim in claims:
        if claim.party_id != party_id:
            continue
        if hints.case_id:
            if claim.case_id == hints.case_id:
                matched.append(claim)
            continue
        parts = claim.created_at.split("-")
        year = parts[0]
        month = int(parts[1])
        if hints.case_type and claim.case_type != hints.case_type:
            continue
        if hints.status and claim.status != hints.status:
            continue
        if hints.year and year != hints.year:
            continue
        if hints.month and month != hints.month:
            continue
        matched.append(claim)
    return matched


def describe_case(claim: Claim) -> str:
    year, month_str, _ = claim.created_at.split("-")
    month_name = calendar.month_name[int(month_str)]
    return f"{claim.case_id} — {claim.case_type}, {month_name} {year}, {claim.status}"


def claim_answer(
    claim: Claim,
    topic: Topic,
    message: str,
    guidance: dict[str, Any],
    now: datetime,
) -> str:
    """Produce a strictly grounded response from the verified claim record and support guidance."""
    docs = claim.documents_needed or []
    documents_formatted = format_list(docs, "and")
    lower_msg = message.lower()
    selected_docs = [doc for doc in docs if doc.lower() in lower_msg]
    relevant_docs = selected_docs if selected_docs else docs

    status_line = f"Claim {claim.case_id} is {claim.status}."
    reason_line = (
        f"The recorded reason is that {claim.denial_reason.rstrip('.')}."
        if claim.denial_reason
        else None
    )
    document_step = (
        f"The requested items are the {documents_formatted}." if documents_formatted else None
    )
    today_iso = now.strftime("%Y-%m-%d")
    expired = bool(claim.appeal_deadline and claim.appeal_deadline < today_iso)
    deadline_line = (
        f"The file lists {claim.appeal_deadline} as the appeal deadline."
        if claim.appeal_deadline
        else None
    )
    expired_step = (
        "That date has passed. A claims representative can review what options may still be available."
    )
    no_documents = "The claim record lists no outstanding document requests."

    def join_parts(*parts: Any) -> str:
        return " ".join(str(p) for p in parts if p)

    def per_document(book_key: str) -> str:
        book = guidance.get(book_key, {})
        snippets: list[str] = []
        for doc in relevant_docs:
            lookup_key = DOCUMENT_KEYS.get(doc, doc)
            entry = book.get(lookup_key) or book.get("default")
            if entry and entry.get("en"):
                snippets.append(entry["en"])
        return " ".join(snippets)

    if topic == Topic.OVERVIEW:
        return join_parts(status_line, reason_line, document_step if not reason_line else None)

    if topic == Topic.DENIAL:
        return join_parts(
            status_line,
            reason_line or "The file does not list a denial reason.",
            deadline_line,
            expired_step if expired else None,
            document_step if not expired else None,
        )

    if topic == Topic.APPEAL_DEADLINE:
        if deadline_line:
            return join_parts(deadline_line, expired_step if expired else None)
        return join_parts(status_line, "The claim record does not list an appeal deadline.")

    if topic == Topic.PAYMENT:
        return join_parts(
            status_line,
            f"The amount paid on record is {format_currency(claim.net_pay)}."
            if claim.net_pay is not None
            else None,
            f"The expected reimbursement on record is {format_currency(claim.expected_reimbursement_amount)}."
            if claim.expected_reimbursement_amount is not None
            else None,
        )

    if topic == Topic.DOCUMENTS:
        if docs:
            return join_parts(document_step, guidance["default_guidance"]["en"])
        return no_documents

    if topic == Topic.ALTERNATIVES:
        if docs:
            return join_parts(
                per_document("document_alternative_guidance"),
                guidance["claim_followup_settings"][
                    "human_review_after_document_alternatives_exhausted"
                ]["en"],
            )
        return no_documents

    if topic in GUIDANCE_TOPICS:
        if not docs:
            return no_documents
        if topic == Topic.SUBMISSION_TIMING and expired:
            return join_parts(deadline_line, expired_step)
        target_topic = GUIDANCE_TOPICS[topic]
        followup_list = guidance.get("claim_followup_guidance", [])
        entry = next((item for item in followup_list if item.get("topic") == target_topic), None)
        if entry:
            avg_time = guidance["claim_followup_settings"][
                "average_processing_time_after_submission"
            ]["en"]
            rendered = (
                entry["en"]
                .replace("{case_id}", claim.case_id)
                .replace("{documents}", documents_formatted)
                .replace("{average_processing_time_after_submission}", avg_time)
            )
            return join_parts(
                rendered,
                per_document("document_guidance") if topic == Topic.FILE_FORMAT else None,
            )

    return (
        "I don’t have that information in the claim record or support guidance. "
        "I can connect you with a claims representative if you’d like."
    )


def summary_for(claim: Claim, discussed: list[str], now: datetime) -> str:
    """Format the post-case email summary body with status, discussed points, and next steps."""
    docs = claim.documents_needed or []
    today_iso = now.strftime("%Y-%m-%d")
    passed = bool(claim.appeal_deadline and claim.appeal_deadline < today_iso)
    if passed:
        next_steps = (
            "Contact claims support to review available options because the recorded appeal deadline has passed."
        )
    elif docs:
        next_steps = (
            "Gather the requested documents and use the member portal or contact claims support for submission guidance."
        )
    else:
        next_steps = (
            "Check the member portal for updates or contact claims support with further questions."
        )

    unique_discussed = list(dict.fromkeys(discussed))
    sections: list[str] = [
        "Here is the summary you requested.",
        f"Claim {claim.case_id} ({claim.case_type}) is {claim.status}.",
    ]
    if unique_discussed:
        bullets = "\n".join(f"• {item}" for item in unique_discussed)
        sections.append(f"What we discussed\n{bullets}")
    if docs:
        sections.append(f"Requested documents\n{format_list(docs, 'and')}.")
    sections.append(f"Next step\n{next_steps}")
    return "\n\n".join(sections)
