const phases = ['VERIFY_ID', 'RESOLVE_INTENT', 'PROCESS_CASE', 'POST_PROCESS'];
const sampleText = 'I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.';
const query = (selector) => document.querySelector(selector);
const messages = query('#messages');
const field = query('#message');
const composer = query('#composer');
const status = query('#status');
const reset = query('#reset');
const sample = query('#sample');
const actions = query('#actions');
const inboxAction = query('#inbox-action');
const stepElements = [...document.querySelectorAll('.phases li')];
const inbox = query('#inbox');
const aiConfig = query('#ai-config');
const configForm = query('#config-form');
let busy = false;
let active = false;
let atStart = true;

function addMessage(text, sender) {
  const entry = document.createElement('div');
  entry.className = `turn ${sender === 'You' ? 'user' : 'assistant'}`;
  const label = document.createElement('p');
  label.className = 'sender';
  label.textContent = sender;
  const bubble = document.createElement('p');
  bubble.className = 'bubble';
  bubble.textContent = text;
  entry.append(label, bubble);
  messages.append(entry);
  messages.scrollTop = messages.scrollHeight;
}

function showStatus(text) {
  status.hidden = !text;
  status.textContent = text;
}

function controls() {
  field.disabled = !active || busy;
  composer.querySelector('button').disabled = !active || busy;
  for (const button of actions.querySelectorAll('button')) button.disabled = busy;
  reset.disabled = busy;
  sample.disabled = busy || !active || !atStart;
}

function render(result) {
  addMessage(result.reply, 'Claims assistant');
  active = result.state === 'active';
  stepElements.forEach((item, index) => {
    const phase = phases[index];
    const current = result.phase === phase && active;
    const complete = result.trace.includes(phase) && (!current && result.phase !== phase || result.state === 'closed');
    if (current) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
    item.classList.toggle('complete', complete);
    item.querySelector('.step').textContent = complete ? '✓' : String(index + 1);
  });
  actions.hidden = !active || result.consent !== 'offered';
  inboxAction.hidden = result.consent !== 'sent';
}

async function request(path, method = 'GET', body) {
  const response = await fetch(`/api/${path}`, {
    method, credentials: 'same-origin',
    ...(body && { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  });
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 401) active = false;
    throw new Error(data.error);
  }
  return data;
}

async function openChat() {
  if (busy) return;
  busy = true;
  controls();
  showStatus('Opening chat…');
  try {
    const result = await request('session', 'POST');
    messages.replaceChildren();
    field.value = '';
    atStart = true;
    query('#mode').textContent = result.modelEnabled ? 'AI interpretation enabled' : 'Local interpretation';
    render(result);
    showStatus('');
  } catch (error) {
    showStatus(error.message);
  } finally {
    busy = false;
    controls();
  }
}

async function post(text) {
  if (busy || !active || !text) return;
  busy = true;
  controls();
  showStatus('Claims assistant is responding…');
  try {
    const result = await request('chat', 'POST', { message: text });
    addMessage(text, 'You');
    field.value = '';
    atStart = false;
    render(result);
    showStatus('');
  } catch (error) {
    showStatus(error.message);
  } finally {
    busy = false;
    controls();
    if (active) field.focus();
  }
}

composer.addEventListener('submit', (event) => {
  event.preventDefault();
  post(field.value.trim());
});
field.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});
sample.addEventListener('click', () => {
  field.value = sampleText;
  field.focus();
});
reset.addEventListener('click', openChat);
query('#send-summary').addEventListener('click', () => post('Send it'));
query('#skip-summary').addEventListener('click', () => post('Skip it'));
query('#open-inbox').addEventListener('click', async () => {
  try {
    const { message } = await request('inbox');
    query('#mail-to').textContent = `To ${message.to}`;
    query('#mail-subject').textContent = message.subject;
    query('#mail-body').textContent = message.body;
    inbox.showModal();
  } catch (error) {
    showStatus(error.message);
  }
});
query('#open-config').addEventListener('click', () => {
  aiConfig.showModal();
});
configForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const result = await request('config', 'POST', {
      token: query('#ai-token').value,
      endpoint: query('#ai-endpoint').value,
      model: query('#ai-model').value,
    });
    query('#mode').textContent = result.modelEnabled ? 'AI interpretation enabled' : 'Local interpretation';
    query('#ai-token').value = '';
    aiConfig.close();
  } catch (error) {
    showStatus(error.message);
  }
});

openChat();
