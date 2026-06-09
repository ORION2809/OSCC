"""One-time OAuth login helper for Google Colab CLI.

The stock Colab CLI tries to open a browser from WSL. On this Windows/WSL
setup that can hang quietly, so this helper prints the authorization URL and
waits for the localhost callback. The saved token path matches Colab CLI.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources

from google_auth_oauthlib.flow import Flow

from colab_cli.auth import OAUTH_SERVER_PORT, PUBLIC_SCOPES, TOKEN_CONFIG_PATH


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None
    error: str | None = None

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        OAuthCallbackHandler.code = params.get("code", [None])[0]
        OAuthCallbackHandler.error = params.get("error", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Colab CLI OAuth complete. You can close this tab.")

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
    client_config = json.loads(config_resource.read_text())

    redirect_uri = f"http://localhost:{OAUTH_SERVER_PORT}/"
    flow = Flow.from_client_config(
        client_config,
        scopes=PUBLIC_SCOPES,
        redirect_uri=redirect_uri,
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    print("Open this URL in your Windows browser:", flush=True)
    print(auth_url, flush=True)

    server = HTTPServer(("localhost", OAUTH_SERVER_PORT), OAuthCallbackHandler)
    server.handle_request()

    if OAuthCallbackHandler.error:
        raise RuntimeError(f"OAuth error: {OAuthCallbackHandler.error}")
    if not OAuthCallbackHandler.code:
        raise RuntimeError("OAuth callback did not include an authorization code.")

    flow.fetch_token(code=OAuthCallbackHandler.code)
    os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
    with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as token_file:
        token_file.write(flow.credentials.to_json())

    print(f"Saved Colab CLI OAuth token: {TOKEN_CONFIG_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
