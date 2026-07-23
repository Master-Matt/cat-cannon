from cat_cannon.app.perception_cache import PerceptionCache


def test_perception_cache_expires_old_bounding_boxes() -> None:
    cache: PerceptionCache[object] = PerceptionCache(max_age_seconds=1.0)
    perception = object()

    cache.update(perception, now=10.0)

    assert cache.current(now=10.999) is perception
    assert cache.current(now=11.001) is None


def test_perception_cache_clears_failed_camera_read() -> None:
    cache: PerceptionCache[object] = PerceptionCache(max_age_seconds=1.0)
    cache.update(object(), now=10.0)

    cache.clear()

    assert cache.current(now=10.1) is None
