from __future__ import annotations

from textual.widgets import Static

from vibe.acp.utils import VibeSessionMode


class ModeIndicator(Static):
    def __init__(self, mode: VibeSessionMode = VibeSessionMode.APPROVAL_REQUIRED) -> None:
        super().__init__()
        self.can_focus = False
        self._mode = mode
        self._update_display()

    def _update_display(self) -> None:
        # Clear all mode classes first
        self.remove_class("mode-approval")
        self.remove_class("mode-edits")
        self.remove_class("mode-auto")

        match self._mode:
            case VibeSessionMode.APPROVAL_REQUIRED:
                self.update("⏵ approval required (shift+tab to toggle)")
                self.add_class("mode-approval")

            case VibeSessionMode.ACCEPT_EDITS:
                self.update("⏵⏵ accept-edits (shift+tab to toggle)")
                self.add_class("mode-edits")

            case VibeSessionMode.AUTO_APPROVE:
                self.update("⏵⏵⏵ auto-approve ALL (shift+tab to toggle)")
                self.add_class("mode-auto")

    def set_mode(self, mode: VibeSessionMode) -> None:
        """Set the current mode and update display."""
        self._mode = mode
        self._update_display()

    def set_auto_approve(self, enabled: bool) -> None:
        """Backward compatibility: convert boolean to mode."""
        self._mode = (
            VibeSessionMode.AUTO_APPROVE if enabled
            else VibeSessionMode.APPROVAL_REQUIRED
        )
        self._update_display()
