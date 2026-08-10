"""The FFmpeg renderer (Phase 7).

Split so the hard part is testable without a binary: :mod:`.graph` builds the filter graph
as pure data, :mod:`.progress` parses FFmpeg's progress stream, and :mod:`.renderer` is the
thin layer that talks to the process.
"""

from app.renderer.ffmpeg.renderer import RENDERER_VERSION, FFmpegRenderer, RenderError

__all__ = ["RENDERER_VERSION", "FFmpegRenderer", "RenderError"]
