import cv2
import numpy as np
import logging

log = logging.getLogger("pose_estimator")

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    log.warning("[Pose] mediapipe not available — posture will default to rule-based")


class PoseEstimator:
    """
    MediaPipe Pose-based posture classifier.
    Estimates standing / crouching / sitting / running from person bbox crops.
    Falls back to bbox aspect-ratio heuristic if MediaPipe unavailable.
    """

    def __init__(self):
        self._pose = None
        if _MP_AVAILABLE:
            try:
                self._mp_pose = mp.solutions.pose
                self._pose = self._mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=0,       # lite model — fast
                    enable_segmentation=False,
                    min_detection_confidence=0.45,
                )
                log.info("[Pose] MediaPipe Pose initialized (lite model)")
            except Exception as e:
                log.warning(f"[Pose] MediaPipe init failed: {e} — using fallback")
                self._pose = None

    def estimate(self, frame: np.ndarray, bbox: list) -> str:
        """
        Args:
            frame: full BGR frame
            bbox:  [x1, y1, x2, y2]
        Returns:
            posture string: "standing" | "crouching" | "sitting" | "running" | "unknown"
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < 20 or box_h < 20:
                return "unknown"

            aspect = box_h / max(box_w, 1)   # tall → standing; wide → sitting

            if self._pose is not None:
                return self._mp_estimate(frame, x1, y1, x2, y2, aspect)
            else:
                return self._heuristic_estimate(aspect)

        except Exception as e:
            log.debug(f"[Pose] estimate error: {e}")
            return "unknown"

    # ------------------------------------------------------------------
    def _mp_estimate(self, frame, x1, y1, x2, y2, aspect) -> str:
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self._pose.process(crop_rgb)

        if not results.pose_landmarks:
            return self._heuristic_estimate(aspect)

        lm = results.pose_landmarks.landmark
        P  = self._mp_pose.PoseLandmark

        def y_(idx): return lm[idx].y        # normalised 0-1 (0=top)
        def vis(idx): return lm[idx].visibility

        # require reasonable visibility for hip + knee
        if vis(P.LEFT_HIP) < 0.3 and vis(P.RIGHT_HIP) < 0.3:
            return self._heuristic_estimate(aspect)

        shoulder_y = (y_(P.LEFT_SHOULDER) + y_(P.RIGHT_SHOULDER)) / 2
        hip_y      = (y_(P.LEFT_HIP)      + y_(P.RIGHT_HIP))      / 2
        knee_y     = (y_(P.LEFT_KNEE)     + y_(P.RIGHT_KNEE))      / 2
        ankle_y    = (y_(P.LEFT_ANKLE)    + y_(P.RIGHT_ANKLE))     / 2

        hip_knee_gap   = knee_y - hip_y      # positive → knee below hip
        knee_ankle_gap = ankle_y - knee_y    # positive → ankle below knee
        torso_height   = hip_y - shoulder_y  # positive → hip below shoulder

        # crouching: knee nearly at hip level AND torso compressed
        if hip_knee_gap < 0.14:
            return "crouching"

        # sitting: hips much lower than expected, knees horizontal with hips
        if hip_knee_gap < 0.22 and aspect < 1.1:
            return "sitting"

        # running: one ankle significantly higher than the other
        left_ankle_y  = y_(P.LEFT_ANKLE)
        right_ankle_y = y_(P.RIGHT_ANKLE)
        ankle_diff = abs(left_ankle_y - right_ankle_y)
        if ankle_diff > 0.18 and hip_knee_gap > 0.20:
            return "running"

        return "standing"

    def _heuristic_estimate(self, aspect: float) -> str:
        """Fallback when no landmarks available."""
        if aspect < 0.85:
            return "sitting"
        elif aspect < 1.2:
            return "crouching"
        else:
            return "standing"

    def close(self):
        if self._pose is not None:
            self._pose.close()
