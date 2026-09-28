"""One-time OAuth per platform. Run on a machine with a browser; copy tokens/ after.

Each flow opens the consent page and catches the redirect on a loopback port. When
the browser cannot reach the port (another machine, a firewall), ``paste=True``
prints the URL and reads the address-bar URL back from stdin instead.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import string
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from scripts.publish.config import TokenStore
from scripts.publish.errors import PublishError
from scripts.publish.platforms import tiktok, youtube

Say = Callable[[str], None]
Ask = Callable[[str], str]

CALLBACK_TIMEOUT = 300.0
FB_SCOPES = "pages_show_list,pages_read_engagement,pages_manage_posts"


@dataclass(frozen=True, slots=True)
class Redirect:
    uri: str
    port: int
    path: str


def _pkce_s256_b64() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _pkce_s256_hex() -> tuple[str, str]:
    """TikTok wants the SHA-256 as hex, not base64url - a documented deviation."""
    alphabet = string.ascii_letters + string.digits + "-._~"
    verifier = "".join(secrets.choice(alphabet) for _ in range(64))
    return verifier, hashlib.sha256(verifier.encode("ascii")).hexdigest()


def _code_from_url(url: str, state: str) -> str:
    query = parse_qs(urlparse(url).query)
    if "error" in query:
        detail = query.get("error_description", query["error"])[0]
        raise PublishError("auth_denied", detail)
    if query.get("state", [""])[0] != state:
        raise PublishError("auth_state_mismatch", "state in the redirect does not match")
    code = query.get("code", [""])[0]
    if not code:
        raise PublishError("auth_no_code", f"no code in {url}")
    return code


def capture_code(
    authorize_url: str,
    redirect: Redirect,
    state: str,
    *,
    paste: bool,
    say: Say,
    ask: Ask,
    open_browser: Callable[[str], bool] = webbrowser.open,
) -> str:
    if paste:
        say(f"Open this URL, approve, then paste the address you land on:\n{authorize_url}")
        return _code_from_url(ask("redirected URL: ").strip(), state)

    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlparse(self.path).path.rstrip("/") != redirect.path.rstrip("/"):
                self.send_response(404)
                self.end_headers()
                return
            captured["url"] = self.path
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h3>Authorised. You can close this tab.</h3>")

        def log_message(self, *_args: Any) -> None:
            return

    server = HTTPServer(("127.0.0.1", redirect.port), Handler)
    server.timeout = 1.0
    thread = threading.Thread(target=_serve_until, args=(server, captured), daemon=True)
    thread.start()
    say(f"Opening the browser. If it does not open, visit:\n{authorize_url}")
    open_browser(authorize_url)
    thread.join(CALLBACK_TIMEOUT)
    server.server_close()
    if "url" not in captured:
        raise PublishError("auth_timeout", "no redirect arrived", hint="retry with --paste")
    return _code_from_url(captured["url"], state)


def _serve_until(server: HTTPServer, captured: dict[str, str]) -> None:
    deadline = time.monotonic() + CALLBACK_TIMEOUT
    while "url" not in captured and time.monotonic() < deadline:
        server.handle_request()


def auth_youtube(
    client: httpx.Client,
    tokens: TokenStore,
    app: dict[str, str],
    *,
    port: int,
    paste: bool,
    say: Say,
    ask: Ask,
) -> dict[str, Any]:
    redirect = Redirect(f"http://127.0.0.1:{port}/", port, "/")
    verifier, challenge = _pkce_s256_b64()
    state = secrets.token_urlsafe(16)
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(
        {
            "client_id": app["client_id"],
            "redirect_uri": redirect.uri,
            "response_type": "code",
            "scope": youtube.SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    code = capture_code(url, redirect, state, paste=paste, say=say, ask=ask)
    response = client.post(
        youtube.TOKEN_URL,
        data={
            "code": code,
            "client_id": app["client_id"],
            "client_secret": app["client_secret"],
            "redirect_uri": redirect.uri,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        },
    )
    data = response.json() if response.status_code == 200 else {}
    if "refresh_token" not in data:
        raise PublishError(
            "auth_exchange_failed",
            f"Google returned no refresh_token: {response.text[:300]}",
            hint="remove the app's access at myaccount.google.com/permissions and retry",
        )
    token = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_at": time.time() + float(data.get("expires_in", 3600)),
        "scope": data.get("scope", youtube.SCOPE),
    }
    tokens.save("youtube", token)
    return {"platform": "youtube", "scope": token["scope"]}


def auth_facebook(
    client: httpx.Client,
    tokens: TokenStore,
    app: dict[str, str],
    limits: dict[str, Any],
    *,
    port: int,
    paste: bool,
    page_id: str | None,
    say: Say,
    ask: Ask,
) -> dict[str, Any]:
    version = limits["facebook"]["graph_version"]
    graph = f"https://graph.facebook.com/{version}"
    redirect = Redirect(f"http://localhost:{port}/callback/", port, "/callback/")
    state = secrets.token_urlsafe(16)
    url = f"https://www.facebook.com/{version}/dialog/oauth?" + urlencode(
        {
            "client_id": app["app_id"],
            "redirect_uri": redirect.uri,
            "state": state,
            "scope": FB_SCOPES,
            "response_type": "code",
        }
    )
    code = capture_code(url, redirect, state, paste=paste, say=say, ask=ask)

    short = client.get(
        f"{graph}/oauth/access_token",
        params={
            "client_id": app["app_id"],
            "client_secret": app["app_secret"],
            "redirect_uri": redirect.uri,
            "code": code,
        },
    )
    if short.status_code != 200:
        raise PublishError("auth_exchange_failed", short.text[:300])
    long_lived = client.get(
        f"{graph}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": app["app_id"],
            "client_secret": app["app_secret"],
            "fb_exchange_token": short.json()["access_token"],
        },
    )
    if long_lived.status_code != 200:
        raise PublishError("auth_exchange_failed", long_lived.text[:300])
    # A Page token derived from a long-lived user token does not expire.
    pages = client.get(
        f"{graph}/me/accounts",
        params={
            "access_token": long_lived.json()["access_token"],
            "fields": "id,name,access_token",
        },
    )
    if pages.status_code != 200:
        raise PublishError("auth_pages_failed", pages.text[:300])
    listed: list[dict[str, str]] = pages.json().get("data", [])
    wanted = page_id or app.get("page_id") or ""
    chosen = [p for p in listed if p["id"] == wanted] if wanted else listed
    if len(chosen) != 1:
        names = ", ".join(f"{p['name']} ({p['id']})" for p in listed) or "none"
        raise PublishError(
            "auth_page_ambiguous",
            f"pages granted: {names}",
            hint="pass --page-id or set page_id in apps.toml",
        )
    page = chosen[0]
    tokens.save(
        "facebook",
        {"page_id": page["id"], "page_name": page["name"], "page_token": page["access_token"]},
    )
    return {"platform": "facebook", "page": page["name"], "page_id": page["id"]}


def auth_tiktok(
    client: httpx.Client,
    tokens: TokenStore,
    app: dict[str, str],
    *,
    port: int,
    paste: bool,
    say: Say,
    ask: Ask,
) -> dict[str, Any]:
    redirect = Redirect(f"http://127.0.0.1:{port}/callback/", port, "/callback/")
    verifier, challenge = _pkce_s256_hex()
    state = secrets.token_urlsafe(16)
    url = (
        tiktok.AUTHORIZE_URL
        + "?"
        + urlencode(
            {
                "client_key": app["client_key"],
                "response_type": "code",
                "scope": tiktok.SCOPE,
                "redirect_uri": redirect.uri,
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
    )
    code = capture_code(url, redirect, state, paste=paste, say=say, ask=ask)
    response = client.post(
        tiktok.TOKEN_URL,
        data={
            "client_key": app["client_key"],
            "client_secret": app["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect.uri,
            "code_verifier": verifier,
        },
    )
    data = response.json() if response.status_code == 200 else {}
    if "access_token" not in data:
        raise PublishError("auth_exchange_failed", response.text[:300])
    if tiktok.SCOPE not in str(data.get("scope", "")):
        raise PublishError(
            "auth_scope_missing",
            f"granted scope is {data.get('scope')!r}",
            hint="enable Content Posting API / video.upload on the app and retry",
        )
    now = time.time()
    tokens.save(
        "tiktok",
        {
            "access_token": data["access_token"],
            "expires_at": now + float(data.get("expires_in", 86400)),
            "refresh_token": data["refresh_token"],
            "refresh_expires_at": now + float(data.get("refresh_expires_in", 0)),
            "open_id": data.get("open_id", ""),
            "scope": data["scope"],
        },
    )
    return {"platform": "tiktok", "scope": data["scope"]}
