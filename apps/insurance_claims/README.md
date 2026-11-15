# Insurance Claims SOP Demo

A Node.js chat app for the supplied insurance fixtures. The sidebar shows progress through identity verification, intent resolution, case processing, and follow-up. Use **Try a sample** to fill Margaret Chen’s message or enter the details over several turns.

## Run locally

Requires Node.js 20 or newer and no package installation.

```sh
npm start
```

Open [the demo on port 3000](http://localhost:3000). The app runs with local interpretation by default. For AI interpretation, set the server-side API token, endpoint, and model and restart:

```sh
export AI_API_TOKEN='<your API token>'
export AI_API_URL='https://api.openai.com/v1/chat/completions'
export AI_MODEL='gpt-4.1-mini'
npm start
```

The endpoint must support bearer authentication and the Chat Completions request and response shape. The example endpoint and model are the defaults. The browser never receives the token. If a provider request fails, times out, or returns an invalid result, the harness uses local interpretation for that turn.

### Docker

Run these commands from this directory. To enable AI interpretation in Docker, export the variables above before running the container.

```sh
docker build -t insurance-claims-sop .
docker run --rm -p 127.0.0.1:3000:3000 \
  -e AI_API_TOKEN -e AI_API_URL -e AI_MODEL insurance-claims-sop
```

## Try the four phases

1. Send: “I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January.”
2. Try asking why it was denied. The assistant should ask for more identity details without opening the claim. Give `DOB is 1985-03-15`, then `SSN last four is 4472`. It should open `CL-2048` using the earlier clue and explain the missing pathology report and office note.
3. Ask “What if I can’t get the pathology report?” to see a grounded answer from the document guidance. For an AI interpretation test, try “How many sleeps between the paperwork landing and you looking at it again?”
4. Choose **Send summary** and open **View demo inbox** to see the recipient, claim status, topics discussed, and next steps. Start a new chat and choose **Skip email** to test the other branch.

To try recovery, say “I already told you who I am. This is ridiculous.” during verification. The assistant explains the privacy requirement and offers other fields. Three consecutive unrelated requests such as “What is RL?” lead to a demo handoff.

## SOP controls

| Phase | Server behavior |
| --- | --- |
| `VERIFY_ID` | Collects distinct factors across turns and checks at least three against one unique fixture record. The policy number is a lookup constraint. Verification failures use a generic response. The assistant remembers case clues while claim access stays blocked. Repeated verification failures, sustained refusals, and proxy callers go to handoff. |
| `RESOLVE_INTENT` | Filters claims by verified party and the caller’s hints. Multiple matches prompt for clarification. An unmatched claim ID produces an account-scoped response. |
| `PROCESS_CASE` | Selects a bounded topic and renders facts from the selected claim and document guidance. The server marks processing complete after its first answer and offers an email summary in the same turn. |
| `POST_PROCESS` | Supports follow-up questions on the selected claim and waits for consent. An explicit send choice delivers one summary to the demo inbox addressed to the fixture email. A skip closes the chat with no email. Questions about email and requests for a different recipient leave consent pending. |

The model sees redacted caller text on eligible turns after verification and returns scope, topic, and emotion classifications. Server code chooses all claim facts, response text, and actions. Verification turns stay local. Sessions are held in memory and expire after 30 minutes. Email delivery and human handoff are simulated. Use the supplied test data when trying the demo.

## Tests

```sh
npm test
```

The tests exercise partial and failed verification, alternate factors, claim hint retention, case isolation and disambiguation, scope and handoff, grounded follow-ups, both consent paths, and the authenticated model request with a stubbed provider.
