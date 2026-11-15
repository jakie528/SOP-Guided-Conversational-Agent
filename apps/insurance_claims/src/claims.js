const monthName = (month) => new Intl.DateTimeFormat('en', { month: 'long', timeZone: 'UTC' })
  .format(new Date(Date.UTC(2000, month - 1)));

export function matchCases(claims, partyId, hints) {
  return claims.filter((claim) => {
    if (claim.party_id !== partyId) return false;
    if (hints.caseId) return claim.case_id === hints.caseId;
    const [year, month] = claim.created_at.split('-');
    return (!hints.caseType || claim.case_type === hints.caseType)
      && (!hints.status || claim.status === hints.status)
      && (!hints.year || year === hints.year)
      && (!hints.month || Number(month) === hints.month);
  });
}

export function describeCase(claim) {
  const [year, month] = claim.created_at.split('-');
  return `${claim.case_id} — ${claim.case_type}, ${monthName(Number(month))} ${year}, ${claim.status}`;
}

const documentKeys = {
  'pathology report': 'original pathology report',
  'office note': 'treating provider office note',
};
const guidanceTopics = {
  submission_method: 'submission_method',
  file_format: 'file_format_requirements',
  submission_timing: 'submission_timing',
  processing_time: 'processing_time_after_submission',
  receipt: 'receipt_confirmation',
};
const currency = (amount) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);

export function claimAnswer(claim, topic, message, guidance, now) {
  const docs = claim.documents_needed ?? [];
  const documents = new Intl.ListFormat('en').format(docs);
  const selectedDocs = docs.filter((doc) => message.toLowerCase().includes(doc));
  const relevantDocs = selectedDocs.length ? selectedDocs : docs;
  const status = `Claim ${claim.case_id} is ${claim.status}.`;
  const reason = claim.denial_reason && `The recorded reason is that ${claim.denial_reason.replace(/\.$/, '')}.`;
  const documentStep = documents && `The requested items are the ${documents}.`;
  const expired = claim.appeal_deadline && claim.appeal_deadline < now.toISOString().slice(0, 10);
  const deadline = claim.appeal_deadline && `The file lists ${claim.appeal_deadline} as the appeal deadline.`;
  const expiredStep = 'That date has passed. A claims representative can review what options may still be available.';
  const noDocuments = 'The claim record lists no outstanding document requests.';
  const answer = (...parts) => parts.filter(Boolean).join(' ');
  const perDocument = (book) => relevantDocs.map((doc) =>
    guidance[book]?.[documentKeys[doc] ?? doc]?.en ?? guidance[book]?.default?.en).filter(Boolean).join(' ');

  if (topic === 'overview') return answer(status, reason, documentStep && !reason && documentStep);
  if (topic === 'denial') return answer(status, reason ?? 'The file does not list a denial reason.', deadline, expired && expiredStep, !expired && documentStep);
  if (topic === 'appeal_deadline') return deadline
    ? answer(deadline, expired && expiredStep)
    : answer(status, 'The claim record does not list an appeal deadline.');
  if (topic === 'payment') return answer(status,
    claim.net_pay !== undefined && `The amount paid on record is ${currency(claim.net_pay)}.`,
    claim.expected_reimbursement_amount !== undefined && `The expected reimbursement on record is ${currency(claim.expected_reimbursement_amount)}.`);
  if (topic === 'documents') return docs.length ? answer(documentStep, guidance.default_guidance.en) : noDocuments;
  if (topic === 'alternatives') return docs.length
    ? answer(perDocument('document_alternative_guidance'), guidance.claim_followup_settings.human_review_after_document_alternatives_exhausted.en)
    : noDocuments;
  if (topic in guidanceTopics) {
    if (!docs.length) return noDocuments;
    if (topic === 'submission_timing' && expired) return answer(deadline, expiredStep);
    const entry = guidance.claim_followup_guidance.find((item) => item.topic === guidanceTopics[topic]);
    return answer(entry.en
      .replaceAll('{case_id}', claim.case_id)
      .replaceAll('{documents}', documents)
      .replaceAll('{average_processing_time_after_submission}', guidance.claim_followup_settings.average_processing_time_after_submission.en),
    topic === 'file_format' && perDocument('document_guidance'));
  }
  return 'I don’t have that information in the claim record or support guidance. I can connect you with a claims representative if you’d like.';
}

export function summaryFor(claim, discussed, now) {
  const docs = claim.documents_needed ?? [];
  const passed = claim.appeal_deadline && claim.appeal_deadline < now.toISOString().slice(0, 10);
  const nextSteps = passed
    ? 'Contact claims support to review available options because the recorded appeal deadline has passed.'
    : docs.length ? 'Gather the requested documents and use the member portal or contact claims support for submission guidance.'
      : 'Check the member portal for updates or contact claims support with further questions.';
  const details = [...new Set(discussed)];
  return [
    'Here is the summary you requested.',
    `Claim ${claim.case_id} (${claim.case_type}) is ${claim.status}.`,
    details.length && `What we discussed\n${details.map((item) => `• ${item}`).join('\n')}`,
    docs.length && `Requested documents\n${new Intl.ListFormat('en').format(docs)}.`,
    `Next step\n${nextSteps}`,
  ].filter(Boolean).join('\n\n');
}
