"""Read messages from a Slack channel using the Web API.

Setup:
    1. Create a Slack app at https://api.slack.com/apps
    2. Add OAuth scopes: channels:history, channels:read, groups:history,
       im:history, mpim:history, users:read
    3. Install the app to your workspace and copy the Bot User OAuth Token
       (starts with "xoxb-") or User OAuth Token (starts with "xoxp-").
    4. Export it: `export SLACK_TOKEN=xoxb-...`
    5. Install the SDK: `pip install slack_sdk`

Usage:
    python slack_reader.py <channel_id> [limit]

Find a channel ID by opening the channel in Slack, clicking the channel name,
and copying the ID at the bottom of the dialog (e.g. C0123ABCD).
"""

import os
import sys
from datetime import datetime

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def fetch_messages(client: WebClient, channel_id: str, limit: int = 20):
    response = client.conversations_history(channel=channel_id, limit=limit)
    return response["messages"]


def resolve_user(client: WebClient, user_id: str, cache: dict) -> str:
    if user_id in cache:
        return cache[user_id]
    try:
        info = client.users_info(user=user_id)
        name = info["user"].get("real_name") or info["user"].get("name") or user_id
    except SlackApiError:
        name = user_id
    cache[user_id] = name
    return name


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python slack_reader.py <channel_id> [limit]", file=sys.stderr)
        return 1

    token = os.environ.get("SLACK_TOKEN")
    if not token:
        print("Set SLACK_TOKEN in your environment.", file=sys.stderr)
        return 1

    channel_id = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    client = WebClient(token=token)
    user_cache: dict[str, str] = {}

    try:
        messages = fetch_messages(client, channel_id, limit)
    except SlackApiError as exc:
        print(f"Slack API error: {exc.response['error']}", file=sys.stderr)
        return 1

    for msg in reversed(messages):
        ts = datetime.fromtimestamp(float(msg["ts"])).strftime("%Y-%m-%d %H:%M:%S")
        user_id = msg.get("user") or msg.get("bot_id") or "unknown"
        author = resolve_user(client, user_id, user_cache) if user_id.startswith("U") else user_id
        text = msg.get("text", "")
        print(f"[{ts}] {author}: {text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
