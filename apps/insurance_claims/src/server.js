import { createServer } from 'node:http';
import { randomBytes } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createHarness, createSession } from './workflow.js';
import { createInterpreter } from './language.js';

const assets = {
  '/': ['index.html', 'text/html; charset=utf-8'],
  '/app.js': ['app.js', 'text/javascript; charset=utf-8'],
  '/style.css': ['style.css', 'text/css; charset=utf-8'],
};

async function fixture(file) {
  return JSON.parse(await readFile(new URL(`../fixtures/${file}.json`, import.meta.url), 'utf8'));
}

const [policyholders, claims, guidance] = await Promise.all([
  fixture('policyholders'), fixture('claims'), fixture('required_document_guideline'),
]);

const token = process.env.AI_API_TOKEN;
const endpoint = process.env.AI_API_URL ?? 'https://api.openai.com/v1/chat/completions';
const model = process.env.AI_MODEL ?? 'gpt-4.1-mini';
if (token) {
  const url = new URL(endpoint);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname))) {
    throw new Error('AI_API_URL must use HTTPS or point to a loopback address.');
  }
}
const harness = createHarness({
  data: { policyholders, claims, guidance }, interpret: createInterpreter({ token, endpoint, model }),
});

const ttl = 30 * 60 * 1000;
const sessions = new Map();
setInterval(() => {
  const cutoff = Date.now() - ttl;
  for (const [key, session] of sessions) if (session.touched < cutoff && !session.busy) sessions.delete(key);
}, 60_000).unref();

function json(res, status, payload) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(payload));
}

class RequestError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function readMessage(req) {
  if (req.headers['content-type']?.split(';')[0] !== 'application/json') throw new RequestError(415, 'Use application/json.');
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 32_768) {
      req.resume();
      throw new RequestError(413, 'Message is too long.');
    }
    chunks.push(chunk);
  }
  let body;
  try {
    body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    throw new RequestError(400, 'Invalid message.');
  }
  if (typeof body?.message !== 'string' || !body.message.trim() || body.message.length > 2500) {
    throw new RequestError(400, 'Enter a message between 1 and 2,500 characters.');
  }
  return body.message.trim();
}

const server = createServer(async (req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
  try {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (req.method === 'GET' && Object.hasOwn(assets, path)) {
      const [file, type] = assets[path];
      const content = await readFile(new URL(`../public/${file}`, import.meta.url));
      res.writeHead(200, { 'Content-Type': type });
      res.end(content);
      return;
    }
    if (!path.startsWith('/api/')) return json(res, 404, { error: 'Not found.' });
    const origin = req.headers.origin;
    if (req.headers['sec-fetch-site'] === 'cross-site'
        || (origin && new URL(origin).host !== req.headers.host)) {
      return json(res, 403, { error: 'Cross-site requests are disabled.' });
    }
    const key = req.headers.cookie?.match(/(?:^|;\s*)claims_session=([a-f\d]{64})(?:;|$)/)?.[1];
    const session = sessions.get(key);
    if (req.method === 'POST' && path === '/api/session') {
      if (session?.busy) return json(res, 409, { error: 'Wait for the current reply before starting again.' });
      if (sessions.size >= 5000 && !session) return json(res, 503, { error: 'The demo is busy. Try again shortly.' });
      if (session) sessions.delete(key);
      const next = randomBytes(32).toString('hex');
      sessions.set(next, { ...createSession(), touched: Date.now() });
      const secure = process.env.SECURE_COOKIE === 'true' ? '; Secure' : '';
      res.setHeader('Set-Cookie', `claims_session=${next}; Path=/; HttpOnly; SameSite=Strict; Max-Age=1800${secure}`);
      return json(res, 200, { ...harness.open(), modelEnabled: Boolean(token) });
    }
    if (!session || Date.now() - session.touched >= ttl) {
      return json(res, 401, { error: 'Your session has expired. Start a new chat to continue.' });
    }
    if (req.method === 'GET' && path === '/api/inbox') {
      if (!session.partyId || session.consent !== 'sent' || !session.mail) {
        return json(res, 404, { error: 'This chat has no delivered summary.' });
      }
      return json(res, 200, { message: session.mail });
    }
    if (req.method === 'POST' && path === '/api/chat') {
      if (session.busy) return json(res, 409, { error: 'Wait for the current reply before sending another message.' });
      session.busy = true;
      try {
        const message = await readMessage(req);
        // Commit a turn atomically after interpretation and workflow processing succeed.
        const next = structuredClone(session);
        const result = await harness.turn(next, message);
        Object.assign(session, next);
        session.touched = Date.now();
        return json(res, 200, result);
      } finally {
        session.busy = false;
      }
    }
    return json(res, 404, { error: 'Not found.' });
  } catch (error) {
    if (!res.headersSent) {
      json(res, error instanceof RequestError ? error.status : 500, {
        error: error instanceof RequestError ? error.message : 'Something went wrong. Please try again.',
      });
    }
  }
});

const port = Number(process.env.PORT ?? 3000);
server.listen(port, process.env.HOST ?? '127.0.0.1', () => {
  console.log(`Claims support demo listening on http://localhost:${server.address().port}`);
});
