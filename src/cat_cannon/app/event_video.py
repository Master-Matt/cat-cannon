from __future__ import annotations

import json
import math
import mimetypes
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from cat_cannon.app.supervisor import SupervisorStepResult
from cat_cannon.config import EventRecordingConfig
from cat_cannon.domain.models import SupervisorState

MIN_PLAYABLE_SECONDS = 1.0


class EventVideoPublisher(Protocol):
    def publish(self, video_path: Path, content: str) -> None:
        ...


class NoopEventVideoPublisher:
    def publish(self, video_path: Path, content: str) -> None:
        return None


@dataclass(frozen=True)
class EventVideoFinalize:
    video_path: Path
    content: str


class TurretEventRecorder:
    def __init__(
        self,
        *,
        config: EventRecordingConfig,
        fps: float,
        publisher: EventVideoPublisher | None = None,
        codec: str = "mp4v",
    ) -> None:
        self.config = config
        self.fps = max(1.0, float(fps))
        self.publisher = publisher or NoopEventVideoPublisher()
        self.codec = codec
        self._writer: Any | None = None
        self._video_path: Path | None = None
        self._frame_size: tuple[int, int] | None = None
        self._started_at: float | None = None
        self._last_shot_at: float | None = None
        self._last_frame_at: float | None = None
        self._last_frame: Any | None = None
        self._frame_remainder = 0.0
        self._frame_count = 0
        self._shot_count = 0
        self._zone_id: str | None = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def update(
        self,
        *,
        cv2: Any,
        turret_frame: Any,
        step_result: SupervisorStepResult,
        now: float | None = None,
    ) -> EventVideoFinalize | None:
        if not self.config.enabled:
            return None

        now_s = time.time() if now is None else float(now)
        cat_in_zone = (
            step_result.active_zone_id is not None
            and step_result.state != SupervisorState.DISARMED
        )
        if not self.is_recording:
            if not cat_in_zone:
                return None
            self._start(
                cv2=cv2,
                turret_frame=turret_frame,
                now=now_s,
                zone_id=step_result.active_zone_id,
            )

        if step_result.active_zone_id is not None:
            self._zone_id = step_result.active_zone_id
        if step_result.fire_commanded:
            self._last_shot_at = now_s
            self._shot_count += 1

        self._write_frame(cv2=cv2, turret_frame=turret_frame, now=now_s)
        if self._should_finalize(cat_in_zone=cat_in_zone, now=now_s):
            return self._finalize(now=now_s)
        return None

    def close(self) -> EventVideoFinalize | None:
        if not self.is_recording:
            return None
        return self._finalize(now=time.time(), reason="shutdown")

    def _start(self, *, cv2: Any, turret_frame: Any, now: float, zone_id: str | None) -> None:
        height, width = turret_frame.shape[:2]
        self._frame_size = (int(width), int(height))
        self._started_at = now
        self._last_shot_at = None
        self._last_frame_at = None
        self._last_frame = None
        self._frame_remainder = 0.0
        self._frame_count = 0
        self._shot_count = 0
        self._zone_id = zone_id
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        self._video_path = output_dir / f"turret_event_{timestamp}_{suffix}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(str(self._video_path), fourcc, self.fps, self._frame_size)
        if hasattr(self._writer, "isOpened") and not self._writer.isOpened():
            self._writer = None
            self._video_path = None
            self._frame_size = None
            raise RuntimeError("failed to open event video writer")

    def _write_frame(self, *, cv2: Any, turret_frame: Any, now: float) -> None:
        if self._writer is None or self._frame_size is None:
            return
        height, width = turret_frame.shape[:2]
        frame = turret_frame
        if (int(width), int(height)) != self._frame_size:
            frame = cv2.resize(turret_frame, self._frame_size)
        frames_to_write = self._frames_due(now)
        self._last_frame = frame
        for _ in range(frames_to_write):
            self._writer.write(frame)
            self._frame_count += 1

    def _frames_due(self, now: float) -> int:
        if self._last_frame_at is None:
            self._last_frame_at = now
            return 1
        elapsed = max(0.0, now - self._last_frame_at)
        self._last_frame_at = now
        exact_frames = elapsed * self.fps + self._frame_remainder
        frames_to_write = int(exact_frames)
        self._frame_remainder = exact_frames - frames_to_write
        return max(0, frames_to_write)

    def _should_finalize(self, *, cat_in_zone: bool, now: float) -> bool:
        if self._started_at is not None and now - self._started_at >= self.config.max_event_seconds:
            return True
        if cat_in_zone:
            return False
        if self._last_shot_at is None:
            return True
        return now - self._last_shot_at >= self.config.post_shot_seconds

    def _finalize(self, *, now: float, reason: str = "complete") -> EventVideoFinalize:
        writer = self._writer
        video_path = self._video_path
        started_at = self._started_at
        last_frame = self._last_frame
        frame_count = self._frame_count
        zone_id = self._zone_id or "-"
        shot_count = self._shot_count
        duration = 0.0 if started_at is None else max(0.0, now - started_at)
        if writer is not None and last_frame is not None:
            min_frames = max(1, int(math.ceil(max(duration, MIN_PLAYABLE_SECONDS) * self.fps)))
            for _ in range(max(0, min_frames - frame_count)):
                writer.write(last_frame)
                frame_count += 1

        self._writer = None
        self._video_path = None
        self._frame_size = None
        self._started_at = None
        self._last_shot_at = None
        self._last_frame_at = None
        self._last_frame = None
        self._frame_remainder = 0.0
        self._frame_count = 0
        self._shot_count = 0
        self._zone_id = None

        if writer is not None:
            writer.release()
        if video_path is None:
            raise RuntimeError("event video finalized without a path")

        content = (
            "Cat Cannon turret event "
            f"zone={zone_id} shots={shot_count} duration={duration:.1f}s reason={reason}"
        )
        self.publisher.publish(video_path, content)
        return EventVideoFinalize(video_path=video_path, content=content)


class DiscordWebhookPublisher:
    def __init__(
        self,
        *,
        webhook_url: str,
        username: str = "Cat Cannon",
        timeout: float = 20.0,
        max_upload_mb: float = 24.0,
        async_publish: bool = True,
        file_ready_timeout: float = 2.0,
        urlopen=urllib.request.urlopen,
    ) -> None:
        self.webhook_url = _validated_discord_webhook_url(webhook_url)
        self.username = username
        self.timeout = float(timeout)
        self.max_upload_bytes = int(max_upload_mb * 1024 * 1024)
        self.async_publish = async_publish
        self.file_ready_timeout = float(file_ready_timeout)
        self._urlopen = urlopen

    def publish(self, video_path: Path, content: str) -> None:
        if not _wait_for_non_empty_file(video_path, timeout=self.file_ready_timeout):
            return
        if video_path.stat().st_size > self.max_upload_bytes:
            return
        if self.async_publish:
            thread = threading.Thread(
                target=self._publish_async,
                args=(video_path, content),
                daemon=True,
            )
            thread.start()
            return
        self._publish_sync(video_path, content)

    def _publish_async(self, video_path: Path, content: str) -> None:
        try:
            self._publish_sync(video_path, content)
        except Exception as exc:
            print(f"[event-video] Discord upload failed: {exc}", flush=True)

    def _publish_sync(self, video_path: Path, content: str) -> None:
        request = _build_discord_file_request(
            webhook_url=self.webhook_url,
            video_path=video_path,
            content=content,
            username=self.username,
        )
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 204)
                if not 200 <= int(status) < 300:
                    raise RuntimeError(f"Discord webhook upload failed with status {status}")
        except urllib.error.URLError as exc:
            raise RuntimeError("Discord webhook upload failed") from exc


def build_event_video_recorder(
    config: EventRecordingConfig,
    *,
    fps: float,
) -> TurretEventRecorder | None:
    if not config.enabled:
        return None
    webhook_url = config.resolved_discord_webhook_url()
    publisher: EventVideoPublisher | None = None
    if webhook_url:
        try:
            publisher = DiscordWebhookPublisher(webhook_url=webhook_url)
        except ValueError as exc:
            print(f"[event-video] Discord upload disabled: {exc}", flush=True)
    return TurretEventRecorder(config=config, fps=fps, publisher=publisher)


def _validated_discord_webhook_url(webhook_url: str) -> str:
    parsed = urllib.parse.urlparse(webhook_url.strip())
    if parsed.scheme != "https":
        raise ValueError("Discord webhook URL must use https")
    hostname = (parsed.hostname or "").lower()
    allowed_hosts = {"discord.com", "discordapp.com"}
    if hostname not in allowed_hosts:
        raise ValueError("Discord webhook URL must point to discord.com")
    if not parsed.path.startswith("/api/webhooks/"):
        raise ValueError("Discord webhook URL must be an /api/webhooks/ URL")
    return urllib.parse.urlunparse(parsed)


def _wait_for_non_empty_file(path: Path, *, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            if path.stat().st_size > 0:
                return True
        except FileNotFoundError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _build_discord_file_request(
    *,
    webhook_url: str,
    video_path: Path,
    content: str,
    username: str,
) -> urllib.request.Request:
    boundary = f"catcannon-{uuid.uuid4().hex}"
    payload = {
        "content": content,
        "username": username,
        "allowed_mentions": {"parse": []},
    }
    body = bytearray()
    _append_form_field(
        body,
        boundary=boundary,
        name="payload_json",
        value=json.dumps(payload),
        content_type="application/json",
    )
    mime_type = mimetypes.guess_type(video_path.name)[0] or "application/octet-stream"
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="files[0]"; '
            f'filename="{video_path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode()
    )
    body.extend(video_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return urllib.request.Request(
        webhook_url,
        data=bytes(body),
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "cat-cannon-event-recorder",
        },
    )


def _append_form_field(
    body: bytearray,
    *,
    boundary: str,
    name: str,
    value: str,
    content_type: str,
) -> None:
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="{name}"\r\n'.encode())
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(value.encode("utf-8"))
    body.extend(b"\r\n")
