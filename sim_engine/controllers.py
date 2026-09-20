import random
import time
from dataclasses import dataclass, field
from typing import Optional


from enum import Enum


class CascadeState(str, Enum):
    SEARCHING = "SEARCHING"
    READY_TO_VERIFY = "READY_TO_VERIFY"
    CONFIRMED = "CONFIRMED"


@dataclass
class CascadeController:
    """
    Adaptive Cascade Decision Controller with 3-state state machine:
    - SEARCHING: LightPackage evaluates frames. Banknote positioning / quality stabilizing.
    - READY_TO_VERIFY: Trigger FullPackage (MQTone + YOLO) for up to max_verify_frames (1-2 runs).
    - CONFIRMED: Note recognized. Stops firing FullPackage, keeping TTS/result active and saving energy.
    - Resets to SEARCHING whenever banknote leaves the frame (has_note=False).
    """
    quality_threshold: float = 0.6
    required_stable_frames: int = 3
    max_verify_frames: int = 2
    _state: CascadeState = field(default=CascadeState.SEARCHING, init=False)
    _stable_count: int = field(default=0, init=False)
    _verify_count: int = field(default=0, init=False)
    _last_decision: bool = field(default=False, init=False)

    def decide(self, has_note: bool, quality: float) -> bool:
        """
        Determines whether to trigger the full detection pipeline.
        Returns:
            True if FullPackage should execute on this frame, False otherwise.
        """
        if not has_note:
            self._state = CascadeState.SEARCHING
            self._stable_count = 0
            self._verify_count = 0
            self._last_decision = False
            return False

        if self._state == CascadeState.CONFIRMED:
            # Already confirmed, hold recognition result and save energy
            self._last_decision = False
            return False

        if self._state == CascadeState.SEARCHING:
            if quality >= self.quality_threshold:
                self._stable_count += 1
                if self._stable_count >= self.required_stable_frames:
                    self._state = CascadeState.READY_TO_VERIFY
                    self._verify_count = 1
                    self._last_decision = True
                    return True
            else:
                self._stable_count = 0
            self._last_decision = False
            return False

        if self._state == CascadeState.READY_TO_VERIFY:
            self._verify_count += 1
            if self._verify_count > self.max_verify_frames:
                # Verification failed to detect a banknote; return to SEARCHING
                self._state = CascadeState.SEARCHING
                self._stable_count = 0
                self._verify_count = 0
                self._last_decision = False
                return False
            self._last_decision = True
            return True

        return False

    def on_detection_result(self, has_detection: bool, conf: float = 0.0):
        """Transitions to CONFIRMED if detection succeeds; resets to SEARCHING if max attempts failed."""
        if has_detection:
            self._state = CascadeState.CONFIRMED
        else:
            if self._verify_count >= self.max_verify_frames:
                self._state = CascadeState.SEARCHING
                self._stable_count = 0
                self._verify_count = 0

    @property
    def state(self) -> CascadeState:
        return self._state

    @property
    def stable_count(self) -> int:
        return self._stable_count

    def reset(self):
        self._state = CascadeState.SEARCHING
        self._stable_count = 0
        self._verify_count = 0
        self._last_decision = False


class SingleShotTrigger:
    """
    Single-Shot Trigger Baseline (B2):
    Simulates traditional camera apps that blindly take a single picture after N seconds
    once a banknote is first spotted.
    N is sampled uniformly from [min_delay_s, max_delay_s] (e.g. 1.5s - 3.0s).
    """
    def __init__(self, min_delay_s: float = 1.5, max_delay_s: float = 3.0, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.trigger_delay = random.uniform(min_delay_s, max_delay_s)
        self._note_first_seen_ts: Optional[float] = None
        self._triggered: bool = False

    def should_trigger(self, has_note: bool, now_ts: float) -> bool:
        """
        Returns True exactly once when delay has elapsed since banknote first appeared.
        """
        if self._triggered:
            return False

        if has_note and self._note_first_seen_ts is None:
            self._note_first_seen_ts = now_ts

        if self._note_first_seen_ts is not None and (now_ts - self._note_first_seen_ts >= self.trigger_delay):
            self._triggered = True
            return True

        return False

    @property
    def has_triggered(self) -> bool:
        return self._triggered

    def reset(self, min_delay_s: float = 1.5, max_delay_s: float = 3.0):
        self.trigger_delay = random.uniform(min_delay_s, max_delay_s)
        self._note_first_seen_ts = None
        self._triggered = False
