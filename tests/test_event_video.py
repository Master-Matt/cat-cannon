import json
from pathlib import Path

from cat_cannon.app.event_video import (
    DiscordWebhookPublisher,
    EventRecordingConfig,
    TurretEventRecorder,
)
from cat_cannon.app.supervisor import SupervisorStepResult
from cat_cannon.domain.models import SupervisorState


class FakeFrame:
    shape = (480, 640, 3)


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

    def VideoWriter_fourcc(self, *codec: str) -> int:
        return 1234

    def VideoWriter(self, path: str, fourcc: int, fps: float, size: tuple[int, int]) -> FakeWriter:
        writer = FakeWriter()
        self.writers.append(writer)
        Path(path).touch()
        return writer


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
) -> SupervisorStepResult:
    return SupervisorStepResult(
        state=state or (SupervisorState.FIRE if fire else SupervisorState.TRACKING),
        fire_commanded=fire,
        human_present=False,
        counter_confirmed=zone is not None,
        target_visible=zone is not None,
        aim_locked=fire,
        active_zone_id=zone,
        candidate_track_id="cat-1" if zone is not None else None,
        correction=None,
    )


def test_turret_event_recorder_records_zone_entry_until_cat_leaves_without_shot(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(enabled=True, output_dir=tmp_path),
        fps=30,
        publisher=publisher,
    )

    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone="counter"), now=0.0)
    recorder.update(cv2=cv2, turret_frame=FakeFrame(), step_result=_result(zone=None), now=0.2)

    assert cv2.writers[0].released is True
    assert len(cv2.writers[0].frames) >= 30
    assert len(publisher.published) == 1
    assert "zone=counter" in publisher.published[0][1]
    assert "shots=0" in publisher.published[0][1]


def test_turret_event_recorder_discards_unconfirmed_short_detection(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=tmp_path,
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


def test_turret_event_recorder_waits_for_zone_lost_hysteresis_after_confirmation(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=tmp_path,
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
    assert len(publisher.published) == 1
    assert "zone=counter" in publisher.published[0][1]
    assert "confirmed=20" in publisher.published[0][1]


def test_turret_event_recorder_ignores_disarmed_zone_detections(tmp_path: Path) -> None:
    cv2 = FakeCv2()
    publisher = FakePublisher()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(enabled=True, output_dir=tmp_path),
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
            output_dir=tmp_path,
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


def test_turret_event_recorder_preserves_wall_clock_duration_for_sparse_updates(
    tmp_path: Path,
) -> None:
    cv2 = FakeCv2()
    recorder = TurretEventRecorder(
        config=EventRecordingConfig(
            enabled=True,
            output_dir=tmp_path,
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


class FakeResponse:
    status = 204

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
