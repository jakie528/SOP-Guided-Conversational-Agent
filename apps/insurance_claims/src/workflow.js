import { extractIdentity, labels, redact, verify } from './identity.js';
import { claimAnswer, describeCase, matchCases, summaryFor } from './claims.js';
import { consentChoice, humanRequested, interpretLocally } from './language.js';

export const phases = ['VERIFY_ID', 'RESOLVE_INTENT', 'PROCESS_CASE', 'POST_PROCESS'];
const greeting = 'Hi, I can help you with an insurance claim. To protect your information, I’ll need to match three identity details before opening a claim. What’s your full name?';
const consentPrompt = 'Would you like me to send a summary to the email address on file? You can say “send it” or “skip it.”';
const empathy = {
  frustrated: 'I hear how frustrating this is.',
  anxious: 'I can hear how worrying this is.',
  confused: 'I can walk through this with you.',
  refusing: 'I understand you may want to keep that detail private.',
};

export function createSession() {
  return {
    phase: phases[0], state: 'active', trace: [phases[0]],
    pendingField: 'name', identity: {}, avoided: [], failures: 0, refusals: 0,
    outOfScope: 0, hints: {}, discussed: [], consent: 'unoffered',
  };
}

function advance(session, phase) {
  if (phases.indexOf(phase) !== phases.indexOf(session.phase) + 1) throw new Error('Invalid phase transition');
  if (!session.partyId) throw new Error('Identity verification required');
  if (phase !== 'RESOLVE_INTENT' && !session.claimId) throw new Error('Claim selection required');
  session.phase = phase;
  session.trace.push(phase);
}

function response(session, reply) {
  return {
    reply,
    phase: session.phase,
    trace: [...session.trace],
    state: session.state,
    consent: session.consent,
  };
}

function handoff(session, reason) {
  session.state = 'handoff';
  session.identity = {};
  if (session.consent === 'offered') session.consent = 'skipped';
  return response(session, `${reason} I’ve requested a handoff to a human claims representative in this demo.`);
}

function remember(session, interpretation) {
  const hints = Object.fromEntries(Object.entries(interpretation.hints).filter(([, value]) => value !== undefined));
  if (hints.caseId) session.hints = { caseId: hints.caseId };
  else if (Object.keys(hints).length) {
    if (session.hints.caseId) session.hints = {};
    Object.assign(session.hints, hints);
  }
  const selectingCase = Object.keys(hints).length > 0 && interpretation.topic === 'overview';
  if (interpretation.topic && (!selectingCase || !session.topic)) session.topic = interpretation.topic;
}

function verifyTurn(session, message, fields, interpreted, data) {
  const lead = empathy[interpreted.emotion];
  if (/\b(?:on behalf of|i.m (?:an? |the )?(?:(?:authorized )?representative|(?:his|her|their) (?:son|daughter|parent|spouse))|not the policyholder)\b/i.test(message)) {
    return handoff(session, 'A representative can verify your identity and authority to act for the policyholder.');
  }

  const askingForAlternate = /\bcan i (?:use|give|share|provide)\b/i.test(message);
  if ((interpreted.emotion === 'refusing' && !askingForAlternate) || /^\s*no[.!]?\s*$/i.test(message)) {
    session.refusals += 1;
    const declined = /\b(?:ssn|social security|national id|last four)\b/i.test(message) ? 'last4'
      : /\b(?:email)\b/i.test(message) ? 'email'
        : /\b(?:phone|telephone)\b/i.test(message) ? 'phone'
          : /\b(?:dob|birthday|birth)\b/i.test(message) ? 'dob' : session.pendingField;
    if (declined && !session.avoided.includes(declined)) session.avoided.push(declined);
  }

  Object.assign(session.identity, fields);
  const holder = verify(session.identity, data.policyholders);
  if (holder) {
    session.partyId = holder.party_id;
    session.identity = {};
    advance(session, 'RESOLVE_INTENT');
    return null;
  }

  const given = Object.keys(session.identity).filter((field) => field in labels);
  if (given.length >= 3) {
    session.failures += 1;
    session.identity = {};
    session.pendingField = 'name';
    if (session.failures >= 3) return handoff(session, 'I wasn’t able to complete verification here.');
    return response(session, 'I couldn’t verify those details together. Let’s try again with three details as they appear on the policy. You can use your full name, date of birth, phone, email, or the last four digits of your SSN or national ID.');
  }
  const available = Object.keys(labels).filter((field) => !given.includes(field) && !session.avoided.includes(field));
  if (session.refusals >= 3 || available.length + given.length < 3) {
    return handoff(session, 'I understand. I’ll keep your claim details protected.');
  }

  const clarifying = askingForAlternate || /\b(?:why|what (?:do|will|can|information)|verify|verification|secure|need my|alternative)\b/i.test(message);
  const protection = 'I need to match three identity details to protect your claim information.';
  let prompt;
  if (given.length || lead || clarifying) {
    session.pendingField = available[0];
    const choices = new Intl.ListFormat('en', { type: 'disjunction' }).format(available.map((field) => labels[field]));
    const count = 3 - given.length;
    prompt = `Please share ${count === 1 ? 'one more detail' : count === 2 ? 'two more details' : 'three details'} using your ${choices}. I can also connect you with a representative.`;
  } else {
    session.pendingField = 'name';
    prompt = 'What’s your full name? You can also start with your date of birth, phone, email, or the last four digits of your SSN or national ID.';
  }
  return response(session, [lead, protection, prompt].filter(Boolean).join(' '));
}

function resolveTurn(session, data) {
  const candidates = matchCases(data.claims, session.partyId, session.hints);
  if (candidates.length === 1) {
    session.claimId = candidates[0].case_id;
    advance(session, 'PROCESS_CASE');
    return null;
  }
  if (candidates.length === 0) {
    const owned = matchCases(data.claims, session.partyId, {});
    session.hints = {};
    if (owned.length === 0) return response(session, 'I don’t see a claim on your verified account. I can connect you with a claims representative for help.');
    return response(session, 'I didn’t find a claim on your account matching those details. Which claim did you mean? You can give me a claim number, type, or month.');
  }
  return response(session, `Which claim would you like to discuss? You can give me its claim number or add a detail about the date or type.\n\n${candidates.map(describeCase).join('\n')}`);
}

export function createHarness({ data, interpret, now = () => new Date() }) {
  return {
    open: () => response(createSession(), greeting),
    async turn(session, message) {
      if (session.state !== 'active') return response(session, 'This chat has ended. Start a new chat if you need help with a claim.');
      if (humanRequested(message)) return handoff(session, 'Of course.');
      const verifying = session.phase === 'VERIFY_ID';
      const { fields, matched } = extractIdentity(message, verifying ? session.pendingField : undefined);
      const safeText = redact(message, matched);
      const local = interpretLocally(safeText);
      const choice = session.phase === 'POST_PROCESS' && session.consent === 'offered' && consentChoice(message);
      if (choice) {
        const claim = matchCases(data.claims, session.partyId, { caseId: session.claimId })[0];
        const holder = data.policyholders.find((item) => item.party_id === session.partyId);
        if (choice === 'send' && claim && holder?.email) {
          session.mail = {
            to: holder.email,
            subject: `Summary for claim ${claim.case_id}`,
            body: summaryFor(claim, session.discussed, now()),
          };
          session.consent = 'sent';
        } else {
          session.consent = 'skipped';
        }
        session.state = 'closed';
        if (session.consent === 'sent') return response(session, 'I sent the summary to the email address on file. You can see it in the demo inbox.');
        if (choice === 'send') return response(session, 'I couldn’t deliver the summary to an email on file. A claims representative can help you with a copy.');
        return response(session, 'I’ll skip the email. Thanks for contacting claims support.');
      }

      // Verification text stays in this process. The optional model has no case or identity data.
      const shorthand = !verifying && session.claimId && /^\s*(?:why (?:was that|is that)|why|what does that mean|what happened)[?.!\s]*$/i.test(message);
      const interpretation = shorthand
        ? { ...local, scope: 'in', topic: 'overview' }
        : verifying ? local : await interpret(safeText);
      const scope = interpretation.scope;
      const lead = empathy[interpretation.emotion];
      const scopeMessage = 'I can help with insurance claims and related support questions.';
      const claimSelection = session.phase === 'RESOLVE_INTENT' && Object.values(local.hints).some(Boolean);
      const identityConversation = verifying && (Object.keys(fields).length > 0 || lead
        || /\b(?:verification|verify|full name|birth|dob|ssn|last four|email|phone|identify|identification|protect|identity|privacy|policyholder)\b/i.test(message));
      const conversational = /^(?:hi|hello|hey|thanks|thank you|got it|okay|ok|yes|no|sure|please)[.!\s]*$/i.test(message.trim());
      const unrelated = scope === 'out' || scope === 'mixed'
        || (scope === 'unknown' && !identityConversation && !claimSelection && !conversational
          && /\?|^\s*(?:what|how|why|when|where|who|which|tell me|explain|define)\b/i.test(safeText));
      if (unrelated) {
        session.outOfScope += 1;
        if (session.outOfScope >= 3) return handoff(session, `${scopeMessage} For further help, please speak with a human representative.`);
        if (scope !== 'mixed') {
          const invitation = session.consent === 'offered' ? consentPrompt
            : verifying ? 'We can continue identity verification whenever you’re ready.'
              : 'What would you like help with on your claim?';
          return response(session, `${scopeMessage} ${invitation}`);
        }
      } else {
        session.outOfScope = 0;
      }

      let prefix = [lead, unrelated && scopeMessage].filter(Boolean).join(' ');
      if (verifying) {
        const caseHint = Object.values(interpretation.hints).some(Boolean);
        const claimQuestion = /\b(?:claim|appeal|reimbursement|denied)\b/i.test(safeText);
        if (scope !== 'out' && (caseHint || claimQuestion)) remember(session, interpretation);
        const result = verifyTurn(session, message, fields, interpretation, data);
        if (result) return unrelated ? { ...result, reply: `${scopeMessage} ${result.reply}` } : result;
        prefix = [prefix, 'Thanks, your identity is verified.'].filter(Boolean).join(' ');
        if (!session.topic && !Object.keys(session.hints).length) {
          return response(session, `${prefix} What would you like help with on your claim?`);
        }
      }

      if (session.phase === 'RESOLVE_INTENT') {
        if (!verifying) {
          if (scope !== 'in' && scope !== 'mixed' && !claimSelection) {
            return response(session, [prefix, 'Tell me what you need help with or give me the claim number, type, or month.'].filter(Boolean).join(' '));
          }
          remember(session, interpretation);
        }
        const result = resolveTurn(session, data);
        if (result) return { ...result, reply: [prefix, result.reply].filter(Boolean).join(' ') };
      }

      const claim = matchCases(data.claims, session.partyId, { caseId: session.claimId })[0];
      if (!claim) return handoff(session, 'I couldn’t open the selected claim.');
      if (!verifying && session.phase === 'POST_PROCESS'
          && /\b(?:different|another|other) claim\b/i.test(message)) {
        return response(session, `We can finish the summary for this claim first, then you can start a new chat for another claim. ${consentPrompt}`);
      }
      if (!verifying && session.phase === 'POST_PROCESS' && Object.values(local.hints).some(Boolean)
          && !matchCases([claim], session.partyId, local.hints).length
          && (local.hints.caseId || local.hints.caseType || local.hints.month && /\bclaim\b/i.test(message))) {
        return response(session, `This chat is about ${claim.case_id}. To discuss another claim, start a new chat or ask for a representative. ${consentPrompt}`);
      }
      if (session.phase === 'POST_PROCESS' && (conversational || /\b(?:email|summary|send it|skip it)\b/i.test(safeText))
          && (scope !== 'in' || !interpretation.topic || interpretation.topic === 'unsupported')) {
        return response(session, [prefix, 'The summary covers your claim status, what we discussed, and next steps. I can send it to the address on file with your permission.', consentPrompt].filter(Boolean).join(' '));
      }

      const topic = session.phase === 'PROCESS_CASE' ? session.topic ?? 'overview' : interpretation.topic ?? 'unsupported';
      const addresseeQuestion = /\b(?:email address|which (?:email|address)|where (?:will|would) (?:it|you) (?:go|send)|send (?:it|the summary) to|different email)\b/i.test(safeText);
      if (session.phase === 'POST_PROCESS' && addresseeQuestion) {
        return response(session, `I can send the summary to the email already on file. A representative can help with address changes. ${consentPrompt}`);
      }
      const answer = claimAnswer(claim, topic, safeText, data.guidance, now());
      if (topic !== 'unsupported' && session.discussed.length < 20) session.discussed.push(answer);
      if (session.phase === 'PROCESS_CASE') {
        advance(session, 'POST_PROCESS');
        session.consent = 'offered';
      }
      return response(session, [prefix, answer, consentPrompt].filter(Boolean).join('\n\n'));
    },
  };
}
