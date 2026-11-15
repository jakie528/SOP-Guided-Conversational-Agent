"""Identity extraction, normalization, PII redaction, and multi-factor verification."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Optional

from .models import IdentityField, Policyholder, Representative

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.? +\d{1,2}(?:st|nd|rd|th)?,? +\d{4}"
    r")\b",
    re.IGNORECASE,
)
PHONE_RE = re.compile(
    r"(?<![\w-])(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?![\w-])"
)
LAST4_RE = re.compile(
    r"\b(?:ssn|social security(?: number)?|national id|last (?:four|4)(?: digits)?)\b"
    r"(?:[\w ]{0,35}?)[:\s#=-]*(\d{4})\b",
    re.IGNORECASE,
)
POLICY_RE = re.compile(r"\bPOL-\d+\b", re.IGNORECASE)
FULL_SSN_RE = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")

EXPLICIT_NAME_LABEL_RE = re.compile(
    r"\b(?:my |full )?name\s*(?:is\s+|[:=]\s*)"
    r"([A-Za-z\u00C0-\u024F'’-]+(?:\s+[A-Za-z\u00C0-\u024F'’-]+){1,4}?)"
    r"(?=$|[,;!?\n]|\.(?:\s|$)|\s+(?:and|policy|dob|email|phone|ssn|born|calling|from|with|about|my)\b)",
    re.IGNORECASE,
)
INTRO_NAME_RE = re.compile(
    r"\b(?:(?:my |full )?name\s*(?:is\s+|[:=]\s*)|(?:i am|i’m|i'm|this is)\s+)"
    r"([A-Z\u00C0-\u00DE][A-Za-z\u00C0-\u024F'’-]+(?:\s+[A-Z\u00C0-\u00DE][A-Za-z\u00C0-\u024F'’-]+){1,4})"
    r"(?=$|[\s,.;!?])",
)
BARE_NAME_WORDS_RE = re.compile(r"^[A-Za-z\u00C0-\u024F.'’ -]+$")
NON_NAME_STOPWORDS_RE = re.compile(
    r"\b(?:the|a|an|policyholder|claim|what|who|why|can|could|how|don’t|don't|do|not|please|no|yes|policy|"
    r"hello|hi|hey|verify|verification|human|agent|representative|good|thank|thanks|"
    r"help|about|am|is|are|was|were|already|refuse|won't|will|tell|explain|define|"
    r"show|describe|skip|send|email|phone|dob|birth|ssn|ridiculous|furious|sad|angry|"
    r"worried|nervous|transfer|connect|capital|france|reinforcement|learning|calling)\b",
    re.IGNORECASE,
)

PROXY_CALLER_RE = re.compile(
    r"\b(?:on behalf of|calling for my (?:mother|father|mom|dad|wife|husband|son|daughter|parent|spouse|client)"
    r"|i['’]?m (?:an? |the )?(?:(?:authorized )?representative|(?:his|her|their) (?:son|daughter|parent|spouse|caregiver|lawyer|attorney))"
    r"|not the policyholder)\b",
    re.IGNORECASE,
)

FIELD_MENTION_PATTERNS: dict[IdentityField, re.Pattern[str]] = {
    IdentityField.NAME: re.compile(r"\b(?:full |my )?name\b", re.IGNORECASE),
    IdentityField.DOB: re.compile(r"\b(?:dob|birthday|birth|date of birth)\b", re.IGNORECASE),
    IdentityField.PHONE: re.compile(r"\b(?:phone|telephone|cell|mobile)\b", re.IGNORECASE),
    IdentityField.EMAIL: re.compile(r"\bemail\b", re.IGNORECASE),
    IdentityField.LAST4: re.compile(
        r"\b(?:ssn|social security|national id|last (?:four|4))\b", re.IGNORECASE
    ),
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_name(value: str) -> str:
    return normalize_text(unicodedata.normalize("NFKC", value))


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def format_date(value: str) -> Optional[str]:
    clean = value.strip()
    numeric = re.match(r"^(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})$", clean)
    if numeric:
        a, b, c = numeric.groups()
        if len(a) == 4:
            year, month, day = int(a), int(b), int(c)
        else:
            year, month, day = int(c), int(a), int(b)
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    # Handle spelled-out month formats like "March 15, 1985" or "Mar. 15th, 1985"
    normalized = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", clean, flags=re.IGNORECASE)
    normalized = normalized.replace(".", "").replace(",", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(normalized, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_identity(
    message: str, expected_field: Optional[IdentityField] = None
) -> tuple[dict[str, str], list[str]]:
    """Extract canonical identity fields and raw matched substrings from a user turn."""
    fields: dict[str, str] = {}
    matched: list[str] = []

    def capture(field_key: str, raw: Optional[str], canonical: Optional[str]) -> None:
        if not raw or not canonical:
            return
        fields[field_key] = canonical
        matched.append(raw)

    # 1. Email
    email_match = EMAIL_RE.search(message)
    if email_match:
        raw_email = email_match.group(0)
        capture("email", raw_email, raw_email.lower())

    # 2. Date of Birth
    dob_match = DATE_RE.search(message)
    if dob_match:
        raw_dob = dob_match.group(0)
        has_dob_cue = bool(re.search(r"\b(?:dob|birth|born|birthday)\b", message, re.IGNORECASE))
        if (
            expected_field == IdentityField.DOB
            or has_dob_cue
            or message.strip().rstrip(".!") == raw_dob
        ):
            capture("dob", raw_dob, format_date(raw_dob))

    # 3. Phone
    phone_match = PHONE_RE.search(message)
    if phone_match:
        raw_phone = phone_match.group(0)
        capture("phone", raw_phone, normalize_phone(raw_phone))

    # 4. SSN / National ID last 4
    last4_match = LAST4_RE.search(message)
    raw_last4: Optional[str] = None
    if last4_match:
        raw_last4 = last4_match.group(1)
    elif expected_field is not None and re.match(r"^\s*\d{4}\s*[.!]?\s*$", message):
        raw_last4 = re.sub(r"\D", "", message)
    capture("last4", raw_last4, raw_last4)

    # 5. Full Name
    full_name: Optional[str] = None
    for pattern in (EXPLICIT_NAME_LABEL_RE, INTRO_NAME_RE):
        for match in pattern.finditer(message):
            candidate = match.group(1).strip()
            if not NON_NAME_STOPWORDS_RE.search(candidate):
                full_name = candidate
                break
        if full_name:
            break

    bare = message.strip().rstrip(".!?")
    if (
        not full_name
        and expected_field == IdentityField.NAME
        and BARE_NAME_WORDS_RE.match(bare)
        and 2 <= len(bare.split()) <= 5
        and not NON_NAME_STOPWORDS_RE.search(bare)
    ):
        full_name = bare

    if full_name:
        capture("name", full_name, normalize_name(full_name))

    # 6. Policy number (lookup constraint only, not counted as one of the 3 required factors)
    policy_match = POLICY_RE.search(message)
    if policy_match:
        raw_policy = policy_match.group(0)
        capture("policy", raw_policy, raw_policy.upper())

    return fields, matched


def redact_pii(message: str, matched: Optional[list[str]] = None) -> str:
    """Redact extracted PII and common sensitive patterns before sending text to external LLMs."""
    safe = message
    for item in matched or []:
        if item:
            safe = safe.replace(item, "[redacted]")
    safe = FULL_SSN_RE.sub("[redacted]", safe)
    safe = EMAIL_RE.sub("[redacted]", safe)
    safe = PHONE_RE.sub("[redacted]", safe)
    safe = DATE_RE.sub("[redacted]", safe)
    safe = POLICY_RE.sub("[redacted]", safe)
    safe = LAST4_RE.sub("[redacted]", safe)
    return safe


def is_proxy_or_representative(
    message: str, representatives: Optional[list[Representative]] = None
) -> bool:
    """Detect whether the caller is an authorized representative or calling on someone else's behalf."""
    if PROXY_CALLER_RE.search(message):
        return True
    if representatives:
        lower_msg = normalize_name(message)
        for rep in representatives:
            rep_norm = normalize_name(rep.rep_name)
            buyer_norm = normalize_name(rep.buyer_name)
            if rep_norm in lower_msg and buyer_norm in lower_msg:
                return True
    return False


def verify_identity(
    fields: dict[str, str], policyholders: list[Policyholder]
) -> Optional[Policyholder]:
    """Verify at least 3 distinct non-policy PII factors against a unique policyholder."""
    factors = [k for k in fields.keys() if k != "policy" and k in {f.value for f in IdentityField}]
    if len(factors) < 3:
        return None

    matches: list[Policyholder] = []
    for holder in policyholders:
        options: dict[str, list[str]] = {
            "name": [normalize_name(holder.name)]
            + [normalize_name(a) for a in holder.name_aliases],
            "dob": [holder.dob],
            "phone": [normalize_phone(holder.phone)]
            + [normalize_phone(a) for a in holder.phone_aliases],
            "email": [normalize_text(holder.email)]
            + [normalize_text(a) for a in holder.email_aliases],
            "last4": [holder.id_last4],
        }
        if "policy" in fields and holder.policy_number != fields["policy"]:
            continue
        if all(fields[f] in options.get(f, []) for f in factors):
            matches.append(holder)

    return matches[0] if len(matches) == 1 else None
