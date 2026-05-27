import json
import urllib.error
from pathlib import Path

from cat_cannon.app.event_video import (
    DiscordWebhookPublisher,
    EventRecordingConfig,
    TurretEventRecorder,
    build_event_video_recorder,
)
from cat_cannon.app.supervisor import SupervisorStepResult
from cat_cannon.domain.models import SupervisorState


class FakeFrame:
    def __init__(self, *, width: int = 640, height: int = 480) -> None:
        self.shape = (height, width, 3)


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[FakeFrame] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: FakeFrame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    def __init__(self) -> None:
        self.writers: list[FakeWriter] = []
        self.writer_sizes: list[tuple[int, int]] = []

    def VideoWriter_fourcc(self, *codec: str) -> int:
        return 1234

    def VideoWriter(self, path: str, fourcc: int, fps: float, size: tuple[int, int]) -> FakeWriter:
        writer = FakeWriter()
        self.writers.append(writer)
        self.writer_sizes.append(size)
        Path(path).touch()
        return writer

    def resize(self, frame: FakeFrame, size: tuple[int, int]) -> FakeFrame:
        width, height = size
        return FakeFrame(width=width, height=height)


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[tuple[Path, str]] = []

    def publish(self, video_path: Path, content: str) -> None:
        self.published.append((video_path, content))


def _result(
    *,
    zone: str | None = "counter",
    fire: bool = False,
    state: SupervisorState | None = None,
    human: bool = False,
    aim_locked: bool | None = None,
    turret_target: bool = False,
    turret_aligned: bool = False,
    turret_direction_aligned: bool = False,
    fire_permitted: bool = False,
    fixed_zone_fresh: bool = True,
) -> SupervisorStepResult:
    return SupervisorStepResult(
        state=state or (SupervisorState.FIRE if fire else SupervisorState.TRACKING),
        fire_commanded=fire,
        human_present=human,
        counter_confirmed=zone is not None,
        target_visible=zone is not None,
        aim_locked=fire if aim_locked is None else aim_locked,
        active_zone_id=zone,
        candidate_track_id="cat-1" if zone is not None else None,
        correction=None,
        turret_target_visible=turret_target,
        turret_fire_aligned=turret_aligned,
        turret_direction_aligned=turret_direction_aligned,
        fire_permitted=fire_permitted,
        fixed_zone_fresh=fixed_zone_fresh,
    )


def test_turret_event_recorder_records_zone_entry_until_cat_leaves_without_shot(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            zone_confirm_detections=1,
            zone_lost_seconds=0.1,
        ),
        fps=30,
        publisher=publisher,
    )

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone="counter"), now=0.0)
    finalized = recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None),
        now=0.2,
    )

    assert finalized is not None
    assert cv2.writers[0].released is True
    assert len(cv2.writers[0].frames) >= 30
    assert publisher.published == []
    assert "zone=counter" in finalized.content
    assert "shots=0" in finalized.content


def test_turret_event_recorder_reports_no_shot_block_reason(tmp_path: Path) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            zone_confirm_detections=1,
            zone_lost_seconds=0.1,
        ),
        fps=30,
        publisher=publisher,
    )

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone="counter", aim_locked=False),
        now=0.0,
    )
    finalized = recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None, state=SupervisorState.HUMAN_LOCKOUT, human=True),
        now=0.2,
    )

    assert publisher.published == []
    assert finalized is not None
    content = finalized.content
    assert "shots=0" in content
    assert "block=human_lockout" in content
    assert "human=1" in content
    assert "aim_locked=0" in content
    assert "states=tracking:1,human_lockout:1" in content


def test_turret_event_recorder_discards_unconfirmed_short_detection(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            zone_confirm_seconds=5.0,
            zone_confirm_detections=20,
            zone_lost_seconds=5.0,
        ),
        fps=10,
        publisher=publisher,
    )

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone="counter"), now=0.0)
    finalized = recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None),
        now=5.1,
    )

    assert finalized is None
    assert publisher.published == []
    assert cv2.writers[0].released is True
    assert not any(tmp_path.glob("*.mp4"))


def test_turret_event_recorder_does_not_confirm_replayed_stale_fixed_zone(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            zone_confirm_seconds=5.0,
            zone_confirm_detections=2,
            zone_lost_seconds=0.1,
        ),
        fps=10,
        publisher=publisher,
    )

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone="counter", fixed_zone_fresh=True),
        now=0.0,
    )
    for index in range(1, 6):
        recorder.update(
            cv2=cv2,
            turret_frame=FakeFrame(),
            step_result=_result(zone="counter", fixed_zone_fresh=False),
            now=index * 0.5,
        )
    finalized = recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None),
        now=5.1,
    )

    assert finalized is None
    assert publisher.published == []
    assert cv2.writers[0].released is True
    assert not any(tmp_path.glob("*.mp4"))


def test_turret_event_recorder_can_publish_no_shot_events_when_configured(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            zone_confirm_detections=1,
            zone_lost_seconds=0.1,
            publish_requires_shot=False,
        ),
        fps=30,
        publisher=publisher,
    )

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone="counter"), now=0.0)
    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=0.2)

    assert len(publisher.published) == 1
    assert "shots=0" in publisher.published[0][1]


def test_turret_event_recorder_close_discards_unconfirmed_detection(tmp_path: Path) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            zone_confirm_seconds=5.0,
            zone_confirm_detections=20,
        ),
        fps=10,
        publisher=publisher,
    )

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone="counter"), now=0.0)

    assert recorder.close() is None
    assert publisher.published == []
    assert cv2.writers[0].released is True
    assert not any(tmp_path.glob("*.mp4"))


def test_turret_event_recorder_waits_for_zone_lost_hysteresis_after_confirmation(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            zone_confirm_seconds=5.0,
            zone_confirm_detections=20,
            zone_lost_seconds=5.0,
        ),
        fps=10,
        publisher=publisher,
    )

    for index in range(20):
        recorder.update(
            cv2=cv2,
            turret_frame=FakeFrame(),
            step_result=_result(zone="counter"),
            now=index * 0.2,
        )

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=4.0)
    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=8.7)

    assert publisher.published == []
    assert cv2.writers[0].released is False

    finalized = recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None),
        now=8.9,
    )

    assert finalized is not None
    assert cv2.writers[0].released is True
    assert publisher.published == []
    assert "zone=counter" in finalized.content
    assert "confirmed=20" in finalized.content


def test_turret_event_recorder_ignores_disarmed_zone_detections(tmp_path: Path) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(enabled=True, output_dir=str(tmp_path)),
        fps=30,
        publisher=publisher,
    )

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone="counter", state=SupervisorState.DISARMED),
        now=0.0,
    )

    assert cv2.writers == []
    assert publisher.published == []


def test_turret_event_recorder_extends_until_post_shot_window(tmp_path: Path) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            post_shot_seconds=15.0,
        ),
        fps=30,
        publisher=publisher,
    )

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone="counter"), now=0.0)
    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone="counter", fire=True),
        now=1.0,
    )
    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=10.0)

    assert publisher.published == []
    assert cv2.writers[0].released is False

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=16.1)

    assert cv2.writers[0].released is True
    assert len(publisher.published) == 1
    assert "shots=1" in publisher.published[0][1]


def test_turret_event_recorder_starts_on_shot_even_if_zone_blinks_out(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            post_shot_seconds=15.0,
        ),
        fps=30,
        publisher=publisher,
    )

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None, fire=True),
        now=1.0,
    )
    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=16.1)

    assert cv2.writers[0].released is True
    assert len(publisher.published) == 1
    assert "zone=-" in publisher.published[0][1]
    assert "shots=1" in publisher.published[0][1]
    assert "block=fired" in publisher.published[0][1]


def test_turret_event_recorder_logs_event_lifecycle(tmp_path: Path, capsys) -> None:
    cv2 = FakeCv2()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            post_shot_seconds=1.0,
            zone_lost_seconds=0.0,
        ),
        fps=30,
    )

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone="counter", fire=True),
        now=1.0,
    )
    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=2.1)

    output = capsys.readouterr().out
    assert "[event-video] started" in output
    assert "reason=shot" in output
    assert "[event-video] finalized" in output
    assert "shots=1" in output
    assert "block=fired" in output


def test_turret_event_recorder_preserves_wall_clock_duration_for_sparse_updates(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            post_shot_seconds=15.0,
        ),
        fps=10,
    )

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone="counter"), now=0.0)
    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone="counter", fire=True),
        now=1.0,
    )
    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=16.1)

    assert cv2.writers[0].released is True
    assert len(cv2.writers[0].frames) >= 160


def test_turret_event_recorder_scales_recorded_frames_to_configured_width(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            max_width=320,
            zone_confirm_detections=1,
            zone_lost_seconds=0.1,
        ),
        fps=10,
    )

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(width=640, height=480),
        step_result=_result(zone="counter"),
        now=0.0,
    )
    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(width=640, height=480),
        step_result=_result(zone=None),
        now=0.2,
    )

    assert cv2.writer_sizes == [(320, 240)]
    assert cv2.writers[0].frames[0].shape == (240, 320, 3)


def test_build_event_video_recorder_uses_configured_fps_and_upload_limit() -> None:
    recorder = build_event_video_recorder(
        EventRecordingConfig(
            enabled=True,
            video_fps=10,
            discord_webhook_url="https://discord.com/api/webhooks/123/token",
            discord_max_upload_mb=8,
        ),
        fps=30,
    )

    assert recorder is not None
    assert recorder.fps == 10
    assert isinstance(recorder.publisher, DiscordWebhookPublisher)
    assert recorder.publisher.max_upload_bytes == 8 * 1024 * 1024


def test_turret_event_recorder_extends_while_turret_target_stays_visible_after_shot(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=str(tmp_path),
            post_shot_seconds=15.0,
            zone_lost_seconds=5.0,
        ),
        fps=10,
        publisher=publisher,
    )

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone="counter", fire=True, turret_target=True, turret_aligned=True),
        now=0.0,
    )
    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None, turret_target=True, turret_aligned=True),
        now=16.0,
    )

    assert publisher.published == []
    assert cv2.writers[0].released is False

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None, turret_target=False, turret_aligned=False),
        now=20.9,
    )
    assert publisher.published == []
    assert cv2.writers[0].released is False

    recorder.update(
        cv2=cv2,
        turret_frame=FakeFrame(),
        step_result=_result(zone=None, turret_target=False, turret_aligned=False),
        now=21.1,
    )

    assert cv2.writers[0].released is True
    assert len(publisher.published) == 1
    assert "shots=1" in publisher.published[0][1]
    assert "turret_target=2" in publisher.published[0][1]


class FakeResponse:
    def __init__(self, status: int = 204) -> None:
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return b""


class FakeUrlopen:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, request, *, timeout: float):
        self.requests.append((request, timeout))
        return FakeResponse()


class FlakyUrlopen:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, request, *, timeout: float):
        self.requests.append((request, timeout))
        if len(self.requests) < 3:
            raise urllib.error.URLError("temporary outage")
        return FakeResponse()


class FakeHttpErrorUrlopen:
    def __call__(self, request, *, timeout: float):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            fp=FakeErrorBody(b'{"message":"bad upload"}'),
        )


class FakeErrorBody:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        return None


def test_discord_webhook_publisher_uploads_video_with_mentions_disabled(tmp_path: Path) -> None:
    video_path = tmp_path / "event.mp4"
    video_path.write_bytes(b"fake-video")
    fake_urlopen = FakeUrlopen()
    publisher = DiscordWebhookPublisher(
        webhook_url="https://discord.com/api/webhooks/123/token",
        urlopen=fake_urlopen,
        async_publish=False,
    )

    publisher.publish(video_path, "cat fired @everyone")

    request, timeout = fake_urlopen.requests[0]
    body = request.data
    assert timeout == 20.0
    assert request.full_url == "https://discord.com/api/webhooks/123/token"
    assert b'name="payload_json"' in body
    assert b'name="files[0]"; filename="event.mp4"' in body
    payload_json = body.split(b"\r\n\r\n", 1)[1].split(b"\r\n", 1)[0]
    payload = json.loads(payload_json.decode("utf-8"))
    assert payload["content"] == "cat fired @everyone"
    assert payload["allowed_mentions"] == {"parse": []}


def test_discord_webhook_publisher_reports_http_failure_body(tmp_path: Path) -> None:
    video_path = tmp_path / "event.mp4"
    video_path.write_bytes(b"fake-video")
    publisher = DiscordWebhookPublisher(
        webhook_url="https://discord.com/api/webhooks/123/token",
        urlopen=FakeHttpErrorUrlopen(),
        async_publish=False,
    )

    try:
        publisher.publish(video_path, "cat fired")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected upload failure")

    assert "status 400" in message
    assert "bad upload" in message


def test_discord_webhook_publisher_retries_transient_upload_failures(tmp_path: Path) -> None:
    video_path = tmp_path / "event.mp4"
    video_path.write_bytes(b"fake-video")
    flaky_urlopen = FlakyUrlopen()
    publisher = DiscordWebhookPublisher(
        webhook_url="https://discord.com/api/webhooks/123/token",
        urlopen=flaky_urlopen,
        async_publish=False,
        retry_attempts=3,
        retry_backoff_seconds=0.0,
    )

    publisher.publish(video_path, "cat fired")

    assert len(flaky_urlopen.requests) == 3


def test_discord_webhook_publisher_skips_empty_video(tmp_path: Path) -> None:
    video_path = tmp_path / "empty.mp4"
    video_path.touch()
    fake_urlopen = FakeUrlopen()
    publisher = DiscordWebhookPublisher(
        webhook_url="https://discord.com/api/webhooks/123/token",
        urlopen=fake_urlopen,
        async_publish=False,
        file_ready_timeout=0.0,
    )

    publisher.publish(video_path, "empty clip")

    assert fake_urlopen.requests == []
