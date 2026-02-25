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
        # Pre-allocate kernel once instead of creating every frame
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame, scale: float = 0.5) -> Tuple[int, any]:
        """
        Processes a frame and checks for motion.
        Internally downscales the frame for faster processing.

        Args:
            frame: The input frame (BGR).
            scale: Scale factor for internal downscaling (default 0.5 = half size).

        Returns:
            Tuple[int, ndarray]: (Non-Zero Pixel Count, Foreground Mask at original resolution)
        """
        # Downscale for faster processing
        if scale < 1.0:
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = frame
        
        # Pre-processing (smaller kernel matches smaller frame)
        blurred = cv2.GaussianBlur(small, (11, 11), 0)
        fgmask_small = self.fgbg.apply(blurred)
        
        # Morphological operations to remove noise
        fgmask_small = cv2.morphologyEx(fgmask_small, cv2.MORPH_OPEN, self.kernel)
        
        # Count movement (on small mask)
        motion_count = cv2.countNonZero(fgmask_small)
        
        # Scale motion count back to approximate original-resolution equivalent
        if scale < 1.0:
            motion_count = int(motion_count / (scale * scale))
            # Upscale mask back to original size for face-motion filtering
            h, w = frame.shape[:2]
            fgmask = cv2.resize(fgmask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            fgmask = fgmask_small
        
        return motion_count, fgmask
