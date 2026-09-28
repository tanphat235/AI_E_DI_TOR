# scripts/publish — post one short to Facebook, YouTube and TikTok

One job folder per video. It holds the video, a `publish.json` with a separate
caption and hashtag set for each platform, and a `ledger.json` recording what has
been posted. A post that half-succeeds can be re-run: platforms already in the
ledger are skipped, only the failed ones are retried.

```
Windows (this machine)                       Linux server (OpenClaw, 24/7)
----------------------                       ------------------------------
publish auth youtube|facebook|tiktok   --->  ~/.aive-publish/{apps.toml,tokens/}   (push-auth)
scripts/publish/                       --->  <root>/scripts/publish/               (push-code)
publish draft clip.mp4                       
  edit publish.json
publish check <job>
publish schedule <job>                 --->  <root>/jobs/<job>/  + an OpenClaw one-shot job
                                             at publish_at:  python -m scripts.publish.cli post jobs/<job>
```

Run every command from the repo root (Windows) or from `<root>` (server):

```powershell
.\.venv\Scripts\python.exe -m scripts.publish.cli <command> ...
```

## What each platform does without an app audit

| platform | what happens | why |
|---|---|---|
| Facebook Page | Reel is published publicly | Standard Access covers Pages your own developer account manages |
| YouTube | uploaded with title, description and tags, then **locked to private** | Google locks uploads from unaudited API projects created after 2020-07-28 ([docs](https://developers.google.com/youtube/v3/docs/videos/insert)) |
| TikTok | video lands in your TikTok **inbox**; you finish the post in the app | inbox upload has no caption field; Direct Post from an unaudited client is `SELF_ONLY` ([docs](https://developers.tiktok.com/doc/content-sharing-guidelines)) |

The `post` summary prints the TikTok caption so it can be pasted, and OpenClaw's
`--announce` delivers that summary to you at post time. TikTok allows at most 5
pending inbox uploads per 24 hours ([docs](https://developers.tiktok.com/doc/content-posting-api-reference-upload-video)).

Both YouTube and TikTok lift their restriction after an audit. Apply once the flow
has run cleanly for a while; neither publishes a turnaround time.

## One-time setup

### 1. App credentials

Copy `apps.example.toml` to `~/.aive-publish/apps.toml` and fill it in as you
create each app below.

### 2. YouTube (Google Cloud)

1. [console.cloud.google.com](https://console.cloud.google.com) → create a project.
2. APIs & Services → Library → enable **YouTube Data API v3**.
3. OAuth consent screen → External → add your Google account under **Test users**.
4. Credentials → Create credentials → OAuth client ID → **Desktop app**. Copy the
   client ID and secret into `[youtube]`.
5. `publish auth youtube` — a browser opens, sign in with the account that owns the channel.

A Desktop client accepts any `http://127.0.0.1:<port>/` redirect, so nothing needs
registering. While the consent screen is in "Testing", Google expires refresh tokens
after 7 days; publish the consent screen (it stays unverified, which is fine for your
own account) to stop that. [Unverified against current Google policy — if `post`
starts failing with `invalid_grant`, re-run `publish auth youtube`.]

### 3. Facebook (Meta app)

1. [developers.facebook.com](https://developers.facebook.com) → My Apps → Create app →
   use case **Manage everything on your Page** (or "Other" → Business).
2. Add product **Facebook Login** → Settings → Valid OAuth Redirect URIs:
   `http://localhost:8765/callback/`
3. App settings → Basic: copy App ID and App Secret into `[facebook]`.
4. `publish auth facebook` — grant `pages_show_list`, `pages_read_engagement`,
   `pages_manage_posts` and tick the Page.

The command trades the login for a long-lived user token and then for a Page token,
which does not expire. If you manage several Pages, put the one you want in
`page_id` or pass `--page-id`.

### 4. TikTok

1. [developers.tiktok.com](https://developers.tiktok.com) → Manage apps → Connect an app.
2. Add products **Login Kit** and **Content Posting API**. Scope: `video.upload`.
3. Login Kit → platform **Desktop** → Redirect URI: `http://127.0.0.1:8765/callback/`
4. Copy Client key and Client secret into `[tiktok]`.
5. `publish auth tiktok`.

Access tokens last 24 hours and are refreshed automatically; the refresh token lasts
a year.

### 5. The server

```bash
# on the server: a venv with two libraries
python3 -m venv ~/aive-publish-venv
~/aive-publish-venv/bin/pip install -r <root>/scripts/publish/requirements.txt
```

```powershell
# from Windows
python -m scripts.publish.cli push-code --host user@server --root /srv/aive-publish
python -m scripts.publish.cli push-auth --host user@server
```

`push-auth` copies `apps.toml` and `tokens/` with `scp`. The tokens are what let the
server post as you: keep the server's `~/.aive-publish` readable by your user only.

## Per video

```powershell
# 1. make a job folder with a copy of the video and a publish.json skeleton
python -m scripts.publish.cli draft projects\x\shorts\clip_001.mp4 --at 2026-10-01T19:00:00+07:00

# 2. fill in publish.json (every "TODO" must go - `check` rejects them)

# 3. validate: lengths, hashtags, time in the future, video duration/fps if PyAV is present
python -m scripts.publish.cli check projects\x\shorts\publish\clip_001

# 4a. post now
python -m scripts.publish.cli post projects\x\shorts\publish\clip_001

# 4b. or schedule it on the server
python -m scripts.publish.cli schedule projects\x\shorts\publish\clip_001 `
    --host user@server --root /srv/aive-publish --python /home/user/aive-publish-venv/bin/python
```

`schedule --dry-run` prints the `scp`, `ssh` and `openclaw automations create`
commands instead of running them. `status <job>` reads the ledger.

## publish.json

```json
{
  "video": "clip_001.mp4",
  "publish_at": "2026-10-01T19:00:00+07:00",
  "platforms": ["facebook", "youtube", "tiktok"],
  "youtube":  {"title": "...", "description": "...", "tags": ["..."], "hashtags": ["#Shorts"]},
  "facebook": {"description": "...", "hashtags": ["#..."]},
  "tiktok":   {"caption": "...", "hashtags": ["#..."]}
}
```

Hashtags are appended to the text on a line of their own. Limits live in
`limits.toml`; the ones not taken from an official doc page are marked there.

## Exit codes

| code | meaning |
|---|---|
| 0 | everything asked for is posted (or checks passed) |
| 2 | configuration problem: missing apps.toml, not authorised, bad arguments |
| 3 | `check` found errors |
| 4 | at least one platform failed; re-run `post` to retry only those |
