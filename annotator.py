import cv2
import datetime
from typing import List, Tuple

class FrameAnnotator:
    """
    Handles drawing annotations (bounding boxes, text) on video frames.
    """

    def draw_faces(self, frame, results: List[Tuple[int, int, int, int, str, any]]):
        """
        Draws bounding boxes and names for detected faces.

        Args:
            frame: The video frame to annotate.
            results: List of (top, right, bottom, left, name, encoding) tuples.
        """
        for (top, right, bottom, left, name, encoding, *_) in results:
            # Color: Red for Unknown, Green for Known
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            # We place the label BELOW the face box
            cv2.rectangle(frame, (left, bottom), (right, bottom + 20), color, cv2.FILLED)
            
            # Draw text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom + 15), font, 0.45, (255, 255, 255), 1)

    def draw_timestamp(self, frame):
        """Draws current timestamp on the frame."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 1)
