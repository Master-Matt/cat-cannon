from __future__ import annotations

from cat_cannon.adapters.interfaces import PerceptionAdapter, PerceptionFrame


class DeepStreamPerceptionAdapter(PerceptionAdapter):
    """Placeholder for the fixed-camera DeepStream pipeline integration."""

    def read_frame(self) -> PerceptionFrame:
        raise NotImplementedError("DeepStream integration is not implemented yet.")

