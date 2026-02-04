import cv2
from typing import Tuple

class MotionDetector:
    """
    Detects motion in video frames using Background Subtraction (MOG2).
    """

    def __init__(self, history: int = 500, threshold: float = 50, detect_shadows: bool = False):
        """
        Args:
            history (int): Length of the history.
            threshold (float): Threshold on the squared Mahalanobis distance.
            detect_shadows (bool): Whether to detect shadows (True) or not (False).
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold, detectShadows=detect_shadows)

    def detect(self, frame: any) -> Tuple[bool, int]:
        """
        Processes a frame and checks for motion.

        Args:
            frame: The input frame (BGR).

        Returns:
            Tuple[bool, int]: (Motion Detected, Non-Zero Pixel Count)
        """
        # Pre-processing
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        fgmask = self.fgbg.apply(blurred)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # Count movement
        motion_count = cv2.countNonZero(fgmask)
        
        return motion_count, fgmask # Returning count mostly, caller decides threshold
