"""FastAPI server for the SOP-Guided Insurance Claims Conversational Agent."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .llm import create_interpreter, validate_endpoint_url
from .models import (
    AIConfigRequest,
    ConsentState,
    FixtureData,
    SessionData,
)
from .workflow import create_harness, create_session

BASE_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BASE_DIR / "fixtures"
PUBLIC_DIR = BASE_DIR / "public"

STATIC_ASSETS: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}

SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 5000
SESSION_COOKIE_RE = re.compile(r"(?:^|;\s*)claims_session=([a-f\d]{64})(?:;|$)")


def load_fixture_data() -> FixtureData:
    """Load and validate all JSON fixtures using Pydantic schemas."""

    def read_json(name: str) -> Any:
        path = FIXTURES_DIR / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    return FixtureData.model_validate(
        {
            "policyholders": read_json("policyholders"),
            "claims": read_json("claims"),
            "guidance": read_json("required_document_guideline"),
            "representatives": read_json("representatives"),
            "claim_schema": read_json("claim_schema"),
            "consent_scenarios": read_json("consent_scenarios"),
        }
    )


def create_app() -> FastAPI:
    fixture_data = load_fixture_data()

    token = (
        os.environ.get("AI_API_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        or os.environ.get("GEMINI_API_KEY")
    )
    is_anthropic = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        or (token and token.startswith("sk-ant-"))
    )
    is_gemini = bool(
        os.environ.get("GEMINI_API_KEY") or (token and token.startswith("AIza"))
    )
    default_url = (
        "https://api.anthropic.com/v1/messages"
        if is_anthropic
        else "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        if is_gemini
        else "https://api.openai.com/v1/chat/completions"
    )
    default_model = (
        "claude-sonnet-4-20250514"
        if is_anthropic
        else "gemini-2.5-flash"
        if is_gemini
        else "gpt-4.1-mini"
    )
    endpoint = os.environ.get("AI_API_URL") or os.environ.get("ANTHROPIC_BASE_URL") or default_url
    model = os.environ.get("AI_MODEL") or os.environ.get("ANTHROPIC_MODEL") or default_model
    if token:
        validate_endpoint_url(endpoint)

    interpreter = create_interpreter(token=token, endpoint=endpoint, model=model)
    harness = create_harness(data=fixture_data, interpret=interpreter)

    sessions: dict[str, SessionData] = {}

    def prune_expired_sessions() -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        expired_keys = [
            k for k, s in sessions.items() if s.touched < cutoff and not s.busy
        ]
        for k in expired_keys:
            sessions.pop(k, None)

    def get_session_key(request: Request) -> str | None:
        cookie_header = request.headers.get("cookie", "")
        match = SESSION_COOKIE_RE.search(cookie_header)
        return match.group(1) if match else None

    app = FastAPI(
        title="Insurance Claims SOP Agent",
        description="SOP-Guided Conversational Agent with Pydantic Structured Outputs and Grounded Emotional Support",
        version="2.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.middleware("http")
    async def security_and_csrf_middleware(request: Request, call_next: Any) -> Response:
        prune_expired_sessions()
        path = request.url.path

        if path.startswith("/api/"):
            sec_fetch_site = request.headers.get("sec-fetch-site")
            origin = request.headers.get("origin")
            host = request.headers.get("host")
            if sec_fetch_site == "cross-site" or (
                origin and urlparse(origin).netloc != host
            ):
                resp = JSONResponse(
                    status_code=403,
                    content={"error": "Cross-site requests are disabled."},
                )
                _apply_security_headers(resp)
                return resp

        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                status_code=500,
                content={"error": "Something went wrong. Please try again."},
            )

        _apply_security_headers(response)
        return response

    def _apply_security_headers(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )

    @app.get("/")
    @app.get("/app.js")
    @app.get("/style.css")
    async def serve_static(request: Request) -> Response:
        asset = STATIC_ASSETS.get(request.url.path)
        if not asset:
            return JSONResponse(status_code=404, content={"error": "Not found."})
        filename, content_type = asset
        content = (PUBLIC_DIR / filename).read_bytes()
        return Response(content=content, media_type=content_type)

    @app.post("/api/session")
    async def open_session(request: Request) -> Response:
        key = get_session_key(request)
        existing = sessions.get(key) if key else None
        if existing and existing.busy:
            return JSONResponse(
                status_code=409,
                content={"error": "Wait for the current reply before starting again."},
            )
        if len(sessions) >= MAX_SESSIONS and not existing:
            return JSONResponse(
                status_code=503,
                content={"error": "The demo is busy. Try again shortly."},
            )

        carry_token = existing.custom_ai_token if existing else None
        carry_url = existing.custom_ai_url if existing else None
        carry_model = existing.custom_ai_model if existing else None
        if key and existing:
            sessions.pop(key, None)

        next_key = secrets.token_hex(32)
        new_session = create_session()
        new_session.touched = time.time()
        new_session.custom_ai_token = carry_token
        new_session.custom_ai_url = carry_url
        new_session.custom_ai_model = carry_model
        sessions[next_key] = new_session

        opened = harness.open()
        payload = {
            **opened.model_dump(mode="json"),
            "modelEnabled": bool(token or carry_token),
        }
        response = JSONResponse(status_code=200, content=payload)
        secure_flag = "; Secure" if os.environ.get("SECURE_COOKIE") == "true" else ""
        response.headers["Set-Cookie"] = (
            f"claims_session={next_key}; Path=/; HttpOnly; SameSite=Strict; Max-Age=1800{secure_flag}"
        )
        return response

    @app.get("/api/inbox")
    async def get_inbox(request: Request) -> Response:
        key = get_session_key(request)
        session = sessions.get(key) if key else None
        if not session or (time.time() - session.touched) >= SESSION_TTL_SECONDS:
            return JSONResponse(
                status_code=401,
                content={"error": "Your session has expired. Start a new chat to continue."},
            )
        if (
            not session.party_id
            or session.consent != ConsentState.SENT
            or not session.mail
        ):
            return JSONResponse(
                status_code=404,
                content={"error": "This chat has no delivered summary."},
            )
        return JSONResponse(
            status_code=200,
            content={"message": session.mail.model_dump(mode="json")},
        )

    @app.post("/api/config")
    async def configure_ai(request: Request) -> Response:
        key = get_session_key(request)
        session = sessions.get(key) if key else None
        if not session or (time.time() - session.touched) >= SESSION_TTL_SECONDS:
            return JSONResponse(
                status_code=401,
                content={"error": "Your session has expired. Start a new chat to continue."},
            )
        try:
            raw_body = await request.json()
            cfg = AIConfigRequest.model_validate(raw_body)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid configuration."})

        if cfg.endpoint:
            try:
                validate_endpoint_url(cfg.endpoint)
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        session.custom_ai_token = cfg.token.strip() if cfg.token and cfg.token.strip() else None
        session.custom_ai_url = cfg.endpoint.strip() if cfg.endpoint and cfg.endpoint.strip() else None
        session.custom_ai_model = cfg.model.strip() if cfg.model and cfg.model.strip() else None
        session.touched = time.time()
        return JSONResponse(
            status_code=200,
            content={"modelEnabled": bool(token or session.custom_ai_token)},
        )

    @app.post("/api/chat")
    async def post_chat(request: Request) -> Response:
        key = get_session_key(request)
        session = sessions.get(key) if key else None
        if not session or (time.time() - session.touched) >= SESSION_TTL_SECONDS:
            return JSONResponse(
                status_code=401,
                content={"error": "Your session has expired. Start a new chat to continue."},
            )
        if session.busy:
            return JSONResponse(
                status_code=409,
                content={"error": "Wait for the current reply before sending another message."},
            )

        session.busy = True
        try:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()
            if content_type != "application/json":
                return JSONResponse(status_code=415, content={"error": "Use application/json."})

            raw_bytes = await request.body()
            if len(raw_bytes) > 32_768:
                return JSONResponse(status_code=413, content={"error": "Message is too long."})

            try:
                body = json.loads(raw_bytes.decode("utf-8"))
            except Exception:
                return JSONResponse(status_code=400, content={"error": "Invalid message."})

            if not isinstance(body, dict) or not isinstance(body.get("message"), str):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Enter a message between 1 and 2,500 characters."},
                )

            raw_msg = body["message"]
            if not raw_msg.strip() or len(raw_msg) > 2500:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Enter a message between 1 and 2,500 characters."},
                )

            message = raw_msg.strip()

            # Atomic turn execution: work on a deep copy and commit only on success
            next_session = session.model_copy(deep=True)
            result = await harness.turn(next_session, message)
            next_session.touched = time.time()
            next_session.busy = False
            sessions[key] = next_session  # type: ignore[index]

            return JSONResponse(
                status_code=200,
                content=result.model_dump(mode="json"),
            )
        finally:
            session.busy = False

    @app.api_route("/api/{rest_of_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def api_not_found(rest_of_path: str) -> Response:
        return JSONResponse(status_code=404, content={"error": "Not found."})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)
