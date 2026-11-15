export const labels = {
  name: 'full name', dob: 'date of birth', phone: 'phone number',
  email: 'email address', last4: 'last four digits of your SSN or national ID',
};

const normalize = (value) => value.trim().toLowerCase().replace(/\s+/g, ' ');
const digits = (value) => value.replace(/\D/g, '');
const phone = (value) => digits(value).replace(/^1(?=\d{10}$)/, '');
const name = (value) => normalize(value.normalize('NFKC'));
const nameWords = /^[\p{L}\p{M}.'’ -]+$/u;

const emailPattern = /[\w.+-]+@[\w.-]+\.[a-z]{2,}/i;
const datePattern = /\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.? +\d{1,2}(?:st|nd|rd|th)?,? +\d{4})\b/i;
const phonePattern = /(?<![\w-])(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?![\w-])/;
const last4Pattern = /\b(?:ssn|social security(?: number)?|national id|last (?:four|4)(?: digits)?)\b(?:[\w ]{0,35}?)[:\s#=-]*(\d{4})\b/i;
const policyPattern = /\bPOL-\d+\b/i;
const namePattern = /\b(?:(?:my |full )?name\s*(?:is\s+|[:=]\s*)|(?:i am|i’m|i'm|this is)\s+)([\p{Lu}][\p{L}\p{M}'’-]+(?:\s+[\p{Lu}][\p{L}\p{M}'’-]+){1,4})(?=$|[\s,.;!?])/u;
const labeledName = /\b(?:my |full )?name\s*(?:is\s+|[:=]\s*)([\p{L}\p{M}'’-]+(?: +[\p{L}\p{M}'’-]+){1,4}?)(?=$|[,;!?\n]|\.(?:\s|$)|\s+(?:and|policy|dob|email|phone|ssn|born|calling)\b)/iu;

function formatDate(value) {
  const numeric = value.match(/^(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})$/);
  let parts;
  if (numeric) {
    const [, a, b, c] = numeric;
    parts = a.length === 4 ? [a, b, c] : [c, a, b];
  } else {
    const clean = value.replace(/(\d)(?:st|nd|rd|th)\b/i, '$1');
    const time = Date.parse(clean + ' UTC');
    if (!Number.isFinite(time)) return undefined;
    return new Date(time).toISOString().slice(0, 10);
  }
  const iso = parts.map((part, i) => part.padStart(i === 0 ? 4 : 2, '0')).join('-');
  const time = Date.parse(iso);
  if (Number.isFinite(time) && new Date(time).toISOString().slice(0, 10) === iso) return iso;
}

export function extractIdentity(message, expectedField) {
  const fields = {};
  const matched = [];
  function capture(field, raw, canonical) {
    if (!raw || !canonical) return;
    fields[field] = canonical;
    matched.push(raw);
  }
  const email = message.match(emailPattern)?.[0];
  capture('email', email, email?.toLowerCase());

  const dob = message.match(datePattern)?.[0];
  if (dob && (expectedField === 'dob' || /\b(?:dob|birth|born|birthday)\b/i.test(message) || message.trim() === dob)) {
    capture('dob', dob, formatDate(dob));
  }
  const telephone = message.match(phonePattern)?.[0];
  capture('phone', telephone, telephone && phone(telephone));
  const idMatch = message.match(last4Pattern);
  const last4 = idMatch?.[1] ?? (expectedField && /^\s*\d{4}\s*$/.test(message) ? message.trim() : undefined);
  capture('last4', last4, last4);

  let fullName = message.match(labeledName)?.[1] ?? message.match(namePattern)?.[1];
  const bare = message.trim().replace(/[.!]$/, '');
  if (!fullName && expectedField === 'name' && /^\p{Lu}/u.test(bare) && nameWords.test(bare)
      && bare.split(/\s+/).length >= 2 && bare.split(/\s+/).length <= 5
      && !/\b(?:claim|what|who|why|can|how|don’t|don't|please|no|policy|hello|verify|human|good|thank|help|about|am|already|refuse|won't|not|tell|explain|define|show|describe)\b/i.test(bare)) {
    fullName = bare;
  }
  capture('name', fullName, fullName && name(fullName));

  const policy = message.match(policyPattern)?.[0];
  capture('policy', policy, policy?.toUpperCase());
  return { fields, matched };
}

export function redact(message, matched = []) {
  let safe = message;
  for (const value of matched) safe = safe.split(value).join('[redacted]');
  return safe
    .replace(/\b\d{3}[- ]?\d{2}[- ]?\d{4}\b/g, '[redacted]')
    .replace(new RegExp(emailPattern.source, 'gi'), '[redacted]')
    .replace(new RegExp(phonePattern.source, 'g'), '[redacted]')
    .replace(new RegExp(datePattern.source, 'gi'), '[redacted]')
    .replace(new RegExp(policyPattern.source, 'gi'), '[redacted]')
    .replace(new RegExp(last4Pattern.source, 'gi'), '[redacted]');
}

export function verify(fields, policyholders) {
  const factors = Object.keys(fields).filter((key) => key !== 'policy');
  if (factors.length < 3) return null;
  const matches = policyholders.filter((holder) => {
    const options = {
      name: [holder.name, ...(holder.name_aliases ?? [])].map(name),
      dob: [holder.dob],
      phone: [holder.phone, ...(holder.phone_aliases ?? [])].map(phone),
      email: [holder.email, ...(holder.email_aliases ?? [])].map(normalize),
      last4: [holder.id_last4],
    };
    return (!fields.policy || holder.policy_number === fields.policy)
      && factors.every((field) => options[field].includes(fields[field]));
  });
  return matches.length === 1 ? matches[0] : null;
}
