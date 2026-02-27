import cv2
import threading
import time
import os
import uuid
import datetime
import logging
from typing import Optional, Dict

import numpy as np
import database
from stream import RTSPStream
from motion import MotionDetector
from annotator import FrameAnnotator
from facial_recognition_system import FacialRecognitionSystem

# Setup Detection Logger
logger = logging.getLogger('detections')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('detections.log')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def ensure_dir_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

class VideoCamera:
    """
    Orchestrates the video pipeline: Frame Capture -> Resize -> Motion -> Face Rec -> Annotation.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.camera_id = config.get('id', 'unknown')
        self.camera_name = config.get('name', 'Unknown Camera')
        
        self.source_url = config.get('video_source', 0)
        self.resolution = tuple(config.get('resolution', [2560, 1920]))
        self.max_fps = int(config.get('max_fps', 10))
        self.face_tolerance = float(config.get('face_match_tolerance', 0.40))
        self.min_face_score = float(config.get('min_face_score', 0.5))
        
        # Display and processing resolutions (separate for performance)
        self.display_width = int(config.get('display_resolution', 800))
        self.process_width = int(config.get('process_resolution', 400))
        self.detection_scale = float(config.get('detection_scale', 0.5))
        
        # 1. Initialize Modules
        self.stream = RTSPStream(self.source_url, self.resolution, max_fps=self.max_fps)
        self.motion_detector = MotionDetector()
        self.fr_system = FacialRecognitionSystem(
            tolerance=self.face_tolerance,
            min_score=self.min_face_score,
            detection_scale=self.detection_scale
        )
        self.annotator = FrameAnnotator()

        # 2. State Management
        self.running = True
        self.last_results = []
        self.jpeg: Optional[bytes] = None
        self.lock = threading.Lock()
        self._viewer_active = False  # Track if anyone is viewing the stream
        self._last_viewer_check = 0.0
        
        # Debounce/Cooldown state
        self.last_logged = {} 
        self.recent_unknowns = [] # Cache for unknown face throttling: [{'id', 'encoding', 'timestamp', 'last_logged'}]
        self.log_cooldown = config.get('log_cooldown', 60)
        
        # Processing settings
        self.process_every_n_frames = int(config.get('process_interval', 5))
        self.frame_count = 0
        self._adaptive_skip = self.process_every_n_frames  # Adaptive skip interval
        self._no_face_streak = 0  # How many scans in a row found no faces
        
        # 3. Start Stream
        self.stream.start()
        
        # 4. Start Processing Loop
        self.thread = threading.Thread(target=self.process_loop, daemon=True)
        self.thread.start()
        
        # Stats tracking
        self.start_time = time.time()
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        print(f"VideoCamera ({self.camera_name}): Orchestration thread started. "
              f"Display: {self.display_width}px, Process: {self.process_width}px, "
              f"Detection scale: {self.detection_scale}")

    def __del__(self):
        self.running = False
        self.stream.stop()
        
    def get_stats(self):
        """Returns current system statistics."""
        uptime_seconds = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"
        
        return {
            "fps": round(self.current_fps, 1),
            "resolution": f"{self.stream.width}x{self.stream.height}",
            "uptime": uptime_str,
            "status": "ONLINE" if self.running and (time.time() - (self.last_frame_time if hasattr(self, 'last_frame_time') else 0) < 5) else "STALLED"
        }

    def process_loop(self):
        """Main processing loop."""
        prev_frame_id = -1
        
        while self.running:
            # 1. Get Frame
            grabbed, frame, frame_id = self.stream.read()
            if not grabbed or frame is None:
                time.sleep(0.1)
                continue
                
            # Skip if we already processed this frame
            if frame_id == prev_frame_id:
                time.sleep(0.01) # Yield
                continue
            
            prev_frame_id = frame_id
            
            # Update FPS Counter
            self.fps_frame_count += 1
            now = time.time()
            if self.fps_frame_count >= 10:
                elapsed = now - self.fps_start_time
                if elapsed > 0:
                    self.current_fps = self.fps_frame_count / elapsed
                self.fps_frame_count = 0
                self.fps_start_time = now
            elif now - self.fps_start_time > 5.0:
                 # Force update if it's been too long (low FPS or stall)
                 elapsed = now - self.fps_start_time
                 if elapsed > 0:
                    self.current_fps = self.fps_frame_count / elapsed
                 self.fps_frame_count = 0
                 self.fps_start_time = now

            # 2. Resize — Separate display and processing frames
            height, width = frame.shape[:2]
            
            # Display frame (larger, for stream viewers)
            if width > self.display_width:
                display_scale = self.display_width / float(width)
                display_h = int(height * display_scale)
                display_frame = cv2.resize(frame, (self.display_width, display_h))
            else:
                display_frame = frame
            
            # Processing frame (smaller, for motion + face detection)
            if width > self.process_width:
                process_scale = self.process_width / float(width)
                process_h = int(height * process_scale)
                process_frame = cv2.resize(frame, (self.process_width, process_h))
            else:
                process_frame = display_frame

            # 3. Motion Detection (on smaller process_frame)
            motion_val, fgmask = self.motion_detector.detect(process_frame)
            motion_threshold = self.config.get('motion_threshold', 10000)
            motion_detected = motion_val > motion_threshold

            # 4. Facial Recognition (Conditional)
            if motion_detected:
                self.frame_count += 1
                if self.frame_count % self._adaptive_skip == 0:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{ts}] [{self.camera_name}] Motion! ({motion_val}) - Scanning Faces...")
                    
                    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                    results = self.fr_system.process_frame(rgb_frame)
                    
                    active_results = []
                    if results:
                        # FILTER: Motion-Guided Face Detection
                        # We only want to process faces that are actually moving.
                        # This filters out static false positives (like parked bicycles).
                        for res in results:
                            # res = (top, right, bottom, left, name, encoding, distance, score)
                            top, right, bottom, left = res[0], res[1], res[2], res[3]
                            
                            # Extract face region from motion mask
                            # Clamp coordinates to fgmask dimensions
                            mh, mw = fgmask.shape[:2]
                            m_top = max(0, min(top, mh))
                            m_bottom = max(0, min(bottom, mh))
                            m_left = max(0, min(left, mw))
                            m_right = max(0, min(right, mw))
                            
                            face_motion_roi = fgmask[m_top:m_bottom, m_left:m_right]
                            
                            # Count moving pixels
                            if face_motion_roi.size > 0:
                                moving_pixels = cv2.countNonZero(face_motion_roi)
                                total_pixels = face_motion_roi.size
                            else:
                                moving_pixels = 0
                                total_pixels = 1
                            
                            # Check if the face has sufficient motion (e.g. > 5% of pixels are moving)
                            if total_pixels > 0:
                                motion_ratio = moving_pixels / total_pixels
                                if motion_ratio > 0.05:
                                    active_results.append(res)
                                else:
                                    print(f"[{ts}] Ignored STATIC face candidate (Motion: {motion_ratio:.2f})")
                            
                        if active_results:
                            print(f"[{ts}] [{self.camera_name}] Faces Found (Moving): {len(active_results)}")
                            self.handle_detections(active_results, process_frame)
                    
                    # Adaptive skip: if no faces found, increase skip to reduce CPU waste
                    if active_results:
                        self._no_face_streak = 0
                        self._adaptive_skip = self.process_every_n_frames  # Reset to normal
                    else:
                        self._no_face_streak += 1
                        if self._no_face_streak >= 3:
                            # After 3 empty scans, double the skip interval (max 4x base)
                            self._adaptive_skip = min(
                                self.process_every_n_frames * 4,
                                self._adaptive_skip * 2
                            )
                    
                    self.last_results = active_results
            else:
                self.last_results = []
                self.frame_count = 0
                self._no_face_streak = 0
                self._adaptive_skip = self.process_every_n_frames  # Reset

            # 5. Annotation (on display frame)
            self.annotator.draw_faces(display_frame, self.last_results)
            # self.annotator.draw_timestamp(display_frame) # Optional

            # 6. Encode for Web Streaming (skip if nobody is watching)
            if self._viewer_active or (now - self._last_viewer_check < 10.0):
                ret, jpeg = cv2.imencode('.jpg', display_frame)
                if ret:
                    with self.lock:
                        self.jpeg = jpeg.tobytes()
                        self.last_frame_time = time.time()
            else:
                # Still update last_frame_time so health checks don't think we stalled
                with self.lock:
                    self.last_frame_time = time.time()
            
            # Yield CPU
            time.sleep(0.01)

    def handle_detections(self, results, frame):
        """Logs detections and fires webhooks."""
        current_time = time.time()
        webhook_url = self.config.get('webhook_url', '')
        
        unknowns_in_frame = []

        for (top, right, bottom, left, name, encoding, distance, score) in results:
            if name == "Unknown":
                # Pass score to unknown handler
                unknowns_in_frame.append((top, right, bottom, left, encoding, distance, score))
                continue

            # Debounce check for Known People
            last_time = self.last_logged.get(name, 0)
            if current_time - last_time > self.log_cooldown:
                self.last_logged[name] = current_time
                
                event_type = "Known Person"
                # Convert distance to generic "Identity Confidence" (inverted)
                # 0.6 is typical threshold. 0.0 is perfect match.
                # Simple inversion: 1.0 - distance.
                identity_conf = max(0.0, 1.0 - distance)
                
                log_text = f"{event_type}: {name} (Identity Confidence: {identity_conf:.2f}, Facial Confidence: {score}) on {self.camera_name}"
                
                # Save snapshot for Review/Correction (add to unknown_faces queue with predicted_name)
                # We do this for EVERY known detection that passes debounce, so we can correct false positives.
                # We save it FIRST so we can get the path for the event log
                snapshot_path = self.save_unknown_snapshot(frame, encoding, (top, right, bottom, left), score, predicted_name=name)
                
                logger.info(f"{log_text}")
                
                # Log to DB with snapshot path
                database.log_event(log_text, snapshot_path, camera_id=self.camera_id)
                
                # Webhook
                if webhook_url:
                    from webhook_manager import send_webhook
                    payload = {
                        "event": "person_detected",
                        "type": event_type,
                        "name": name,
                        "timestamp": current_time,
                        "camera": self.camera_name,
                        "camera_id": self.camera_id,
                        "identity_confidence": identity_conf,
                        "facial_confidence": score,
                        "snapshot_path": snapshot_path
                    }
                    send_webhook(webhook_url, payload)

        # Handle Unknowns Batch
        if unknowns_in_frame:
            # Smart Unknown Throttling
            # Instead of a global timer, we check if this specific unknown face has been seen recently.
            
            # Find best unknown face (highest score) to potentially log
            best_unknown = max(unknowns_in_frame, key=lambda x: x[6])
            
            # Check if ANY of the unknowns in frame are "active" (not on cooldown)
            # We treat them individually for cooldowns, but might preserve batch logging for cleaner logs?
            # Actually, per-person logging is better for specific timestamps/snapshots.
            
            for (top, right, bottom, left, encoding, distance, score) in unknowns_in_frame:
                
                # Check against recent unknowns cache
                matched_id = None
                is_new = True
                
                # Cleanup old cache entries first (remove entries not seen for 2x cooldown)
                min_time = current_time - (self.log_cooldown * 2)
                self.recent_unknowns = [u for u in self.recent_unknowns if u['last_seen'] > min_time]
                
                for u in self.recent_unknowns:
                    # Cosine distance (matches facial_recognition_system.py logic)
                    # ArcFace embeddings are L2-normalized, so dot product = cosine similarity
                    similarity = np.dot(u['encoding'], encoding)
                    cosine_dist = 1.0 - similarity
                    
                    if cosine_dist < self.face_tolerance:
                        matched_id = u['id']
                        is_new = False
                        # Keep the cache entry alive while they're still in frame
                        u['last_seen'] = current_time
                        u['encoding'] = encoding  # Update with latest encoding
                        
                        if current_time - u['last_logged'] > self.log_cooldown:
                            # Re-log this person after cooldown expired
                            u['last_logged'] = current_time
                            u['best_score'] = score
                            self.log_unknown_event(frame, encoding, (top, right, bottom, left), score, u['id'], is_relog=True)
                        elif score > u.get('best_score', 0):
                            # On cooldown but better snapshot available — silently upgrade
                            u['best_score'] = score
                            self._upgrade_snapshot(u.get('snapshot_path'), frame, encoding, (top, right, bottom, left), score)
                        break
                
                if is_new:
                    # Create new temporary ID
                    new_id = str(uuid.uuid4())[:8]
                    snapshot_path = self.log_unknown_event(frame, encoding, (top, right, bottom, left), score, new_id, is_relog=False)
                    self.recent_unknowns.append({
                        'id': new_id,
                        'encoding': encoding,
                        'last_seen': current_time,    # For cache cleanup
                        'last_logged': current_time,  # For cooldown
                        'best_score': score,
                        'snapshot_path': snapshot_path
                    })

    def log_unknown_event(self, frame, encoding, location, score, temp_id, is_relog=False):
        """Helper to log unknown event. Returns snapshot path."""
        log_text = f"Unknown Person (ID: {temp_id}, Score: {score}) on {self.camera_name}"
        if is_relog:
            log_text += " [Recurring]"
            
        logger.info(f"{log_text}")
        
        snapshot_path = self.save_unknown_snapshot(frame, encoding, location, score)
        database.log_event(log_text, snapshot_path, camera_id=self.camera_id)
        
        # Webhook
        webhook_url = self.config.get('webhook_url', '')
        if webhook_url:
            from webhook_manager import send_webhook
            payload = {
                "event": "person_detected",
                "type": "Unknown Person",
                "name": "Unknown",
                "id": temp_id,
                "timestamp": time.time(),
                "camera": self.camera_name,
                "camera_id": self.camera_id,
                "facial_confidence": score,
                "snapshot_path": snapshot_path
            }
            send_webhook(webhook_url, payload)
        
        return snapshot_path

    def _upgrade_snapshot(self, snapshot_path, frame, encoding, location, score):
        """Silently upgrades an existing snapshot with a better-quality capture."""
        if not snapshot_path or not os.path.exists(snapshot_path):
            return
        
        top, right, bottom, left = location
        height, width = frame.shape[:2]
        pad = 50
        
        new_top = max(0, top - pad)
        new_bottom = min(height, bottom + pad)
        new_left = max(0, left - pad)
        new_right = min(width, right + pad)
        
        cropped_face = frame[new_top:new_bottom, new_left:new_right]
        cv2.imwrite(snapshot_path, cropped_face)
        
        logging.debug(f"Upgraded snapshot {os.path.basename(snapshot_path)} (score: {score:.2f})")

    def save_unknown_snapshot(self, frame, encoding, location, score=0.0, predicted_name=None):
        """Saves a snapshot of unknown (or known) faces, cropped to the face."""
        snapshot_dir = self.config.get('snapshot_dir', 'static/snapshots')
        ensure_dir_exists(snapshot_dir)
        filename = f"unknown_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(snapshot_dir, filename)
        
        # Crop Frame with Padding
        top, right, bottom, left = location
        height, width = frame.shape[:2]
        pad = 50 # padding in pixels
        
        # Calculate new coordinates with clamping
        new_top = max(0, top - pad)
        new_bottom = min(height, bottom + pad)
        new_left = max(0, left - pad)
        new_right = min(width, right + pad)
        
        cropped_face = frame[new_top:new_bottom, new_left:new_right]
        
        cv2.imwrite(filepath, cropped_face)
        database.add_unknown_face(encoding, filepath, camera_id=self.camera_id, detection_score=score, predicted_name=predicted_name)
        
        return filepath

    def get_frame(self) -> Optional[bytes]:
        """Returns the current JPEG frame bytes. Returns None if stale."""
        # Signal that a viewer is active (enables JPEG encoding)
        self._viewer_active = True
        self._last_viewer_check = time.time()
        
        with self.lock:
            # Check for staleness using last_frame_time (if it exists)
            # If we haven't received a frame in > 5 seconds, consider it stalled
            # and return None to signal the app/UI.
            last_time = getattr(self, 'last_frame_time', 0)
            if time.time() - last_time > 5.0:
                return None
                
            return self.jpeg
        
    def get_last_frame_time(self):
        return self.last_frame_time if hasattr(self, 'last_frame_time') else 0
