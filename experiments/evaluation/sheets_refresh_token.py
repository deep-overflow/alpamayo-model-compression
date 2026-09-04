"""Mint a long-lived Google refresh token once, so pushing results stops needing a browser.

The sheets MCP connector's OAuth flow omits `access_type=offline`, so Google returns only
a one-hour access token and no refresh token -- which is why the cached credential in
`.credentials.json` has an `accessToken` but no `refreshToken`, and why every push after
the first hour asks for a reconnect.

This runs the same authorisation against the same OAuth client, with `access_type=offline`
and `prompt=consent` added, which is exactly what makes Google issue a refresh token. That
token does not expire on a timer (only after ~6 months unused, or if revoked), and
`push_to_sheet.py` exchanges it for a fresh access token on every run.

The client id and secret are read from the connector's own entry in the Claude Code
credential store, so nothing new has to be registered in Google Cloud.

  # 1. print the URL to authorise (once)
  .venv/bin/python experiments/evaluation/sheets_refresh_token.py auth

  # 2. paste back the address-bar URL from the redirect, which fails to load by design --
  #    port 8765 has no listener here, but the code is in the URL
  .venv/bin/python experiments/evaluation/sheets_refresh_token.py exchange "<pasted url>"

Writes `~/.config/sheets_refresh.json` (0600). After that `push_to_sheet.py` runs
unattended; `--token-file` and $SHEETS_ACCESS_TOKEN still take precedence.
"""

import base64
import hashlib
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN = "https://oauth2.googleapis.com/token"
REDIRECT = "http://localhost:8765/callback"
SCOPES = ("https://www.googleapis.com/auth/drive "
          "https://www.googleapis.com/auth/spreadsheets")
# The sheets connector's OAuth client id. Not a secret: it is printed verbatim in the
# authorisation URL the connector hands out. Only the matching secret is confidential,
# and that is read from the credential store rather than written here.
DEFAULT_CLIENT_ID = ("671919272234-2o0jt64n22vodns5f4oik0u0sgsrhpca"
                     ".apps.googleusercontent.com")
STORE = Path(os.environ.get("SHEETS_REFRESH_STORE",
                            Path.home() / ".config" / "sheets_refresh.json"))
# the verifier has to survive between the two invocations, so it is written beside the
# store rather than held in memory
PENDING = STORE.with_suffix(".pending.json")


def claude_store():
    cands = []
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        cands.append(Path(os.environ["CLAUDE_CONFIG_DIR"]) / ".credentials.json")
    cands += [Path.home() / "project/chan/.claude/.credentials.json",
              Path.home() / ".claude/.credentials.json"]
    return [p for p in cands if p.exists()]


def client_creds():
    """(client_id, client_secret) from the connector's own cached registration.

    The id is not secret -- it appears in the authorisation URL the connector prints --
    and the secret sits next to it under mcpOAuthClientConfig.
    """
    for p in claude_store():
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for key, v in (d.get("mcpOAuthClientConfig") or {}).items():
            if key.startswith("sheets|") and v.get("clientSecret"):
                # the store keeps only the secret, so the id falls back to the
                # published constant
                cid = (v.get("clientId") or os.environ.get("SHEETS_CLIENT_ID")
                       or DEFAULT_CLIENT_ID)
                return cid, v["clientSecret"]
    raise SystemExit("no sheets OAuth client in the Claude Code credential store; "
                     "connect the sheets connector once (/mcp) first.")


def post_form(url, fields):
    body = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{url} -> HTTP {e.code}: {e.read().decode()[:400]}") from None


def cmd_auth():
    cid, _ = client_creds()
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    state = secrets.token_urlsafe(24)
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(json.dumps({"verifier": verifier, "state": state}))
    PENDING.chmod(0o600)
    q = urllib.parse.urlencode({
        "response_type": "code", "client_id": cid, "redirect_uri": REDIRECT,
        "scope": SCOPES, "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
        # the two parameters the connector's flow leaves out, and the whole point here
        "access_type": "offline", "prompt": "consent",
    })
    print("Open this once, authorise, then paste the address-bar URL into `exchange`:\n")
    print(f"{AUTH}?{q}\n")
    print("The redirect page will fail to load (nothing listens on 8765) -- that is fine, "
          "the code is in the URL.")


def cmd_exchange(url):
    if not PENDING.exists():
        raise SystemExit("no pending flow; run `auth` first")
    pend = json.loads(PENDING.read_text())
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "code" not in qs:
        raise SystemExit(f"no ?code= in that URL. Got keys: {sorted(qs)}")
    if qs.get("state", [None])[0] != pend["state"]:
        raise SystemExit("state mismatch -- the URL is from a different flow; re-run `auth`")
    cid, secret = client_creds()
    tok = post_form(TOKEN, {
        "grant_type": "authorization_code", "code": qs["code"][0],
        "client_id": cid, "client_secret": secret,
        "redirect_uri": REDIRECT, "code_verifier": pend["verifier"],
    })
    if "refresh_token" not in tok:
        raise SystemExit(
            "Google returned no refresh_token. That happens when the account has already "
            "granted this client offline access; revoke it at "
            "https://myaccount.google.com/permissions and retry, or re-run `auth` (it "
            "already sends prompt=consent).")
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps({"client_id": cid, "client_secret": secret,
                                 "refresh_token": tok["refresh_token"]}, indent=2))
    STORE.chmod(0o600)
    PENDING.unlink(missing_ok=True)
    print(f"refresh token stored in {STORE} (0600).")
    print("push_to_sheet.py now runs without a browser.")


def access_token():
    """Exchange the stored refresh token for a fresh access token. Used by push_to_sheet."""
    if not STORE.exists():
        return None
    d = json.loads(STORE.read_text())
    tok = post_form(TOKEN, {
        "grant_type": "refresh_token", "refresh_token": d["refresh_token"],
        "client_id": d["client_id"], "client_secret": d["client_secret"],
    })
    return tok["access_token"]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("auth", "exchange", "test"):
        raise SystemExit(f"usage: {sys.argv[0]} {{auth|exchange <url>|test}}")
    if sys.argv[1] == "auth":
        cmd_auth()
    elif sys.argv[1] == "exchange":
        if len(sys.argv) < 3:
            raise SystemExit("paste the callback URL as the second argument")
        cmd_exchange(sys.argv[2])
    else:
        t = access_token()
        print("no stored refresh token" if not t
              else f"minted an access token, {len(t)} chars -- the stored grant works")


if __name__ == "__main__":
    main()
