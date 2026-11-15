import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createHarness, createSession, phases } from '../src/workflow.js';
import { createInterpreter } from '../src/language.js';

const fixture = async (file) => JSON.parse(await readFile(new URL(`../fixtures/${file}.json`, import.meta.url)));
const data = {
  policyholders: await fixture('policyholders'),
  claims: await fixture('claims'),
  guidance: await fixture('required_document_guideline'),
};
const now = () => new Date('2026-11-15T01:00:00.000Z');
const harness = createHarness({ data, now, interpret: createInterpreter({}) });
const sample = 'I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.';
const stagedStart = 'My name is Margaret Chen. I’m calling about my denied healthcare claim from January.';

test('sample traverses all four phases with three matches and grounded denial details', async () => {
  const session = createSession();
  const result = await harness.turn(session, sample);
  assert.deepEqual(result.trace, phases);
  assert.match(result.reply, /verified[\s\S]*CL-2048[\s\S]*pathology report[\s\S]*office note[\s\S]*email address on file/);
  assert.equal(result.consent, 'offered');
  assert.deepEqual(session.identity, {});
  assert.equal(session.mail, undefined);
});

test('the early case hint survives partial identity, frustrations, and replacement fields', async () => {
  const session = createSession();
  const first = await harness.turn(session, stagedStart);
  assert.equal(first.phase, 'VERIFY_ID');
  assert.doesNotMatch(first.reply, /CL-2048|pathology report/);
  const frustrated = await harness.turn(session, 'I already told you who I am. This is ridiculous. Just tell me why my claim was denied.');
  assert.match(frustrated.reply, /frustrating[\s\S]*protect[\s\S]*two more details/);
  assert.equal(frustrated.phase, 'VERIFY_ID');
  await harness.turn(session, 'I’d rather not give my SSN.');
  const second = await harness.turn(session, 'My DOB is March 15, 1985.');
  assert.equal(second.phase, 'VERIFY_ID');
  assert.doesNotMatch(second.reply, /last four|pathology report/);
  const final = await harness.turn(session, 'Phone is (650) 521-2836');
  assert.equal(final.phase, 'POST_PROCESS');
  assert.match(final.reply, /CL-2048/);
});

test('policy number is no substitute for the third matched field, and mismatches reveal no claim', async () => {
  const session = createSession();
  assert.equal((await harness.turn(session, 'My name is Margaret Chen, POL-9921. DOB is 1985-03-15.')).phase, 'VERIFY_ID');
  const wrong = await harness.turn(session, 'SSN last four is 9999');
  assert.equal(wrong.phase, 'VERIFY_ID');
  assert.match(wrong.reply, /couldn’t verify those details together/);
  assert.doesNotMatch(wrong.reply, /claim CL-|Margaret|date of birth matched/);
});

test('alias and national ID are supported; representatives are referred to a human', async () => {
  const session = createSession();
  await harness.turn(session, 'My name is Yaven Li. Email is yawen.li@example.com and national ID last four is 5317.');
  assert.equal(session.partyId, 'P13');
  const proxy = await harness.turn(createSession(), 'My name is David Chen and I’m calling on behalf of Margaret Chen.');
  assert.equal(proxy.state, 'handoff');
  assert.equal(proxy.phase, 'VERIFY_ID');
});

test('verified lookup is scoped to the holder and asks about ambiguous matches', async () => {
  const session = createSession();
  const result = await harness.turn(session, sample.replace('my denied healthcare claim from January', 'my healthcare claim from January'));
  assert.equal(result.phase, 'RESOLVE_INTENT');
  assert.match(result.reply, /CL-2048[\s\S]*CL-2011/);
  const otherAccount = await harness.turn(session, 'CL-3001');
  assert.match(otherAccount.reply, /didn’t find a claim on your account/);
  assert.doesNotMatch(otherAccount.reply, /diagnosis report/);
  const selected = await harness.turn(session, 'CL-2011');
  assert.equal(selected.phase, 'POST_PROCESS');
  assert.match(selected.reply, /CL-2011 is closed/);
});

test('rejects unrelated questions and requests human handoff after repeated retries', async () => {
  const session = createSession();
  for (const message of ['What is RL?', 'Explain reinforcement learning.', 'What is the capital of France?']) {
    const result = await harness.turn(session, message);
    assert.match(result.reply, /insurance claims/);
    assert.doesNotMatch(result.reply, /CL-2048|pathology report/);
  }
  assert.equal(session.state, 'handoff');
});

test('answers follow-ups from the fixture and sends a single summary after explicit consent', async () => {
  const session = createSession();
  await harness.turn(session, sample);
  const followup = await harness.turn(session, 'What if I cannot get the original pathology report?');
  assert.match(followup.reply, /request|replacement copy/);
  assert.equal(session.mail, undefined);
  for (const message of ['Could you send it?', 'Yes, send it to alternate@example.com']) {
    const pending = await harness.turn(session, message);
    assert.equal(pending.consent, 'offered');
    assert.equal(session.mail, undefined);
  }
  const done = await harness.turn(session, 'Yes, please send it');
  assert.equal(done.consent, 'sent');
  assert.equal(done.state, 'closed');
  assert.equal(session.mail.to, 'margaret@email.com');
  assert.match(session.mail.body, /CL-2048[\s\S]*denied[\s\S]*replacement copy[\s\S]*Next step/);
  assert.doesNotMatch(session.mail.body, /4472|1985-03-15|POL-9921|alternate@example.com/);
  const original = session.mail;
  await harness.turn(session, 'Send it');
  assert.equal(session.mail, original);
});

test('skip is terminal and advance consent is ignored', async () => {
  const session = createSession();
  assert.equal((await harness.turn(session, 'Yes, please send it')).consent, 'unoffered');
  await harness.turn(session, sample);
  const result = await harness.turn(session, 'No thanks');
  assert.equal(result.consent, 'skipped');
  assert.equal(result.state, 'closed');
  await harness.turn(session, 'Yes');
  assert.equal(session.mail, undefined);
});

test('AI can map messy language onto a bounded topic without access to verification text or records', async () => {
  const calls = [];
  const interpret = createInterpreter({
    token: 'test-token', endpoint: 'https://model.invalid/v1/chat/completions', model: 'test-model',
    fetchImpl: async (url, request) => {
      const body = JSON.parse(request.body);
      calls.push({ url, headers: request.headers, body });
      return Response.json({ choices: [{ message: { content: JSON.stringify({
        scope: 'in', topic: 'processing_time', emotion: 'anxious', authorization: 'send', answer: 'made-up answer',
      }) } }] });
    },
  });
  const withModel = createHarness({ data, now, interpret });
  const session = createSession();
  await withModel.turn(session, sample);
  assert.equal(calls.length, 0);
  const result = await withModel.turn(session, 'How many sleeps between the paperwork landing and you looking at it again?');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].headers.Authorization, 'Bearer test-token');
  assert.doesNotMatch(JSON.stringify(calls[0].body), /Margaret|4472|CL-2048|pathology report/);
  assert.match(result.reply, /worrying[\s\S]*usually less than a week/);
  assert.doesNotMatch(result.reply, /made-up answer/);
  assert.equal(result.consent, 'offered');
});

test('provider failure falls back to local interpretation', async () => {
  const interpret = createInterpreter({
    token: 'test-token', endpoint: 'https://model.invalid', model: 'test-model',
    fetchImpl: async () => new Response(null, { status: 503 }),
  });
  assert.equal((await interpret('How do I upload the pathology report?')).topic, 'submission_method');
});
