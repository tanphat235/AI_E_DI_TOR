"""Container-level facts about a media file, as reported by ``ffprobe``.

This lives apart from :mod:`app.models.video` and :mod:`app.models.audio` because
both need it and neither owns it. A probe is a measurement of a file, made once
and cached; everything downstream treats it as ground truth about what the file
actually contains, as opposed to what a plan claims it contains.
"""

from __future__ import annotations

from pydantic import Field

from app.models.common import AiveModel, MediaRef, TimeRange


class VideoStreamInfo(AiveModel):
    """The properties of a file's primary video stream."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0, description="Average frame rate, frames per second.")
    codec: str = Field(min_length=1, description="ffprobe codec_name, e.g. 'h264'.")
    pixel_format: str | None = None
    bit_rate: int | None = Field(default=None, gt=0, description="Bits per second.")
    rotation: int = Field(
        default=0,
        description=(
            "Display rotation in degrees from container metadata. Phone footage is "
            "routinely stored landscape with a 90-degree rotation flag, so analysis "
            "must use the *display* orientation, not the encoded one."
        ),
    )

    @property
    def display_size(self) -> tuple[int, int]:
        """Width and height after applying container rotation."""
        if self.rotation % 180 == 90:
            return self.height, self.width
        return self.width, self.height

    @property
    def aspect(self) -> float:
        """Display aspect ratio, width over height."""
        width, height = self.display_size
        return width / height


class AudioStreamInfo(AiveModel):
    """The properties of a file's primary audio stream."""

    codec: str = Field(min_length=1)
    sample_rate: int = Field(gt=0, description="Samples per second.")
    channels: int = Field(gt=0)
    bit_rate: int | None = Field(default=None, gt=0)


class MediaProbe(AiveModel):
    """Everything AIVE needs to know about a media container.

    Both streams are optional: a narration file has audio only, and silent B-roll
    has video only. Code that needs one must check, which is deliberate - assuming
    every clip carries audio is a classic source of renderer failures.
    """

    source: MediaRef
    duration: float = Field(gt=0.0, description="Container duration in seconds.")
    size_bytes: int = Field(ge=0)
    format_name: str = Field(default="", description="ffprobe format_name.")
    video: VideoStreamInfo | None = None
    audio: AudioStreamInfo | None = None

    @property
    def has_video(self) -> bool:
        return self.video is not None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @property
    def full_range(self) -> TimeRange:
        """The whole file as a time range, for clamping plan-supplied ranges."""
        return TimeRange(start=0.0, end=self.duration)


__all__ = ["AudioStreamInfo", "MediaProbe", "VideoStreamInfo"]
