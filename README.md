# Insurance Claims SOP Demo (Python + FastAPI + Pydantic v2)

An SOP-guided conversational agent built with **Python 3.11+**, **FastAPI**, and **Pydantic v2**. The harness enforces strict business workflow gates (`VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS`) while using LLM structured outputs (`TurnAnalysis`) and guardrailed empathetic response synthesis (`compose_empathy` + `validate_empathy_guardrail`) across phases.

## Run Locally

Create a virtual environment and install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.server
```

Open [http://localhost:3000](http://localhost:3000) (interactive OpenAPI docs are also available at [http://localhost:3000/api/docs](http://localhost:3000/api/docs)).

## Configure Any AI Model / Provider

You can pass an API token either by clicking **API token** in the web UI header or via environment variables.

If you only set `AI_API_TOKEN`, the server automatically detects the provider from the key prefix (`sk-...` for OpenAI, `sk-ant-...` for Anthropic Claude, `AIza...` for Google Gemini) and selects the matching endpoint and default model:

```sh
export AI_API_TOKEN='<your API token>'
python -m src.server
```

You can also explicitly specify `AI_API_URL` and `AI_MODEL` for any provider:

* **OpenAI (default)**
  ```sh
  export AI_API_TOKEN='<your-openai-key>'
  export AI_API_URL='https://api.openai.com/v1/chat/completions'
  export AI_MODEL='gpt-4.1-mini'
  python -m src.server
  ```
* **Anthropic Claude**
  ```sh
  export AI_API_TOKEN='<your-anthropic-key>'
  export AI_API_URL='https://api.anthropic.com/v1/messages'
  export AI_MODEL='claude-sonnet-4-20250514'
  python -m src.server
  ```
* **Google Gemini**
  ```sh
  export AI_API_TOKEN='<your-gemini-key>'
  export AI_API_URL='https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
  export AI_MODEL='gemini-2.5-flash'
  python -m src.server
  ```
* **Any OpenAI-Compatible Provider (OpenRouter, Groq, DeepSeek, Mistral, Together, xAI, Ollama/vLLM)**
  ```sh
  export AI_API_TOKEN='<your-provider-key>'
  export AI_API_URL='https://openrouter.ai/api/v1/chat/completions' # or Groq / DeepSeek / Mistral / localhost Ollama URL
  export AI_MODEL='<provider-model-name>'
  python -m src.server
  ```

If no token is configured or if a provider request fails/times out, the harness automatically falls back to deterministic local interpretation for that turn.

### Docker

```sh
docker build -t insurance-claims-sop .
docker run --rm -p 127.0.0.1:3000:3000 \
  -e AI_API_TOKEN -e AI_API_URL -e AI_MODEL insurance-claims-sop
```

## Architecture Overview

| Module | Responsibility |
| --- | --- |
| `src/models.py` | Pydantic v2 models for fixture validation (`Policyholder`, `Claim`, `Representative`), SOP state (`SessionData`, `Phase`, `ConsentState`), and LLM structured outputs (`TurnAnalysis`, `CaseHints`). |
| `src/identity.py` | Deterministic PII extraction, Unicode/date/phone normalization, 3-factor verification (`verify_identity`), representative detection (`representatives.json`), and strict PII redaction (`redact_pii`). |
| `src/claims.py` | Account-scoped claim filtering (`match_cases`), grounded answer synthesis from `claims.json` and `required_document_guideline.json`, and post-case email summary generation. |
| `src/llm.py` | Async `httpx` LLM client supporting OpenAI, Anthropic Claude, and Google Gemini, producing validated `TurnAnalysis` structured outputs and `validate_empathy_guardrail` so LLM-generated empathy never leaks claim details or bypasses SOP gates. |
| `src/workflow.py` | `SOPHarness` state machine enforcing `VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS`, cross-phase hint retention, emotional de-escalation, and 3-strike human handoff escalation. |
| `src/server.py` | FastAPI server with atomic session commits, CSP/CSRF protection, static UI hosting, and runtime API token configuration (`/api/config`). |

## Run Tests

```sh
pytest -v
```
