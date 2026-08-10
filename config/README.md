# Site-level configuration overrides

The **canonical defaults** ship inside the package at
[app/config/default.toml](../app/config/default.toml). That file is the reference
for every tunable value in AIVE and is the one to read when you want to know what
a setting does.

Do not edit the packaged file. Override it instead, at whichever level fits:

| Level | File | Use it for |
|---|---|---|
| Packaged defaults | `app/config/default.toml` | reference only — do not edit |
| Site override | `config/aive.toml` (this folder) | machine-wide choices: GPU device, ffmpeg path |
| Project override | `<project>/aive.toml` | per-video choices: aspect ratio, pacing, subtitle style |
| Environment | `AIVE_<SECTION>__<FIELD>` | CI and one-off runs |
| CLI flag | e.g. `--fps 24` | a single invocation |

Later entries win. Overrides are merged per field, so a file needs to contain only
what it changes:

```toml
# config/aive.toml — this machine has a GPU and a system ffmpeg build
[speech]
device = "cuda"
compute_type = "float16"

[media]
ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"
```

Environment variables use a double underscore to descend into a section:

```powershell
$env:AIVE_RULES__MIN_CLIP_DURATION = "2.0"
$env:AIVE_OUTPUT__FPS = "24"
```

Run `aive config show` to see the merged result and where each value came from.
