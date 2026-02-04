import cv2
import threading
import time
import subprocess
import numpy as np
import logging
import queue
import json
from typing import Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class RTSPStream:
    """
    Handles the connection to the RTSP stream using a direct FFmpeg pipe.
    This bypasses OpenCV's internal backend issues and enforces TCP + Timeouts reliably.
    """

    def __init__(self, source: str, resolution: Optional[Tuple[int, int]] = None, max_fps: int = 30):
        self.source = source
        self.max_fps = max_fps
        self.resolution = resolution # Optional manual override
        self.running = False
        self.threading = False
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.frame_id = 0
        self.grabbed: bool = False
        self.thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen] = None
        
        # Stream properties
        if resolution and resolution != (0, 0):
            self.width, self.height = resolution
        else:
            self.width = 0
            self.height = 0

    def start(self) -> 'RTSPStream':
        """Starts the frame reading thread."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Stream: Connecting via FFmpeg Pipe (Max FPS: {self.max_fps})...")
        self.running = True
        self.last_bw_check = time.time() # Reset time
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        """Stops the frame reading thread and releases resources."""
        self.running = False
        if self.thread:
            self.thread.join()
        self._close_process()

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        with self.lock:
            return self.grabbed, self.frame, self.frame_id

    def _close_process(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            self.process = None

    def _probe_resolution(self):
        """Uses ffprobe to determine stream resolution."""
        try:
            # Use shell=True with single quotes around URL to handle special chars like $
            # This is safer for passwords with symbols
            cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 -rtsp_transport tcp '{self.source}'"
            
            logger.info(f"Probing: {cmd}")
            output = subprocess.check_output(cmd, shell=True, timeout=10).decode().strip()
            
            if not output:
                return None, None
                
            w, h = map(int, output.split('x'))
            logger.info(f"Stream Resolution detected: {w}x{h}")
            return w, h
        except Exception as e:
            print(f"Probe failed: {e}. Retrying resolution detection...")
            return None, None

    def _update(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Stream: Thread started.")
        retry_delay = 2
        
        while self.running:
            # 1. Probe resolution if needed
            if self.width == 0:
                print("Stream: Probing resolution...")
                w, h = self._probe_resolution()
                if w and h:
                    self.width, self.height = w, h
                    print(f"Stream: Resolution found: {w}x{h}")
                else:
                    print(f"Stream: Resolution probe failed. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                    continue

            # 2. Start FFmpeg
            # Robust RTSP/Camera flags
            cmd = [
                'ffmpeg',
                '-loglevel', 'error',
                '-rtsp_transport', 'tcp',
                '-timeout', '5000000',
                '-probesize', '32M',          # Analyze more data to find streams
                '-analyzeduration', '10M',    # Spend more time analyzing inputs
                '-fflags', '+genpts+discardcorrupt', # Discard bad packets
                '-i', self.source,
                '-f', 'image2pipe',
                '-pix_fmt', 'bgr24',
                '-vcodec', 'rawvideo',
                '-r', str(self.max_fps),
                '-'
            ]
            
            print(f"Stream: Launching FFmpeg...")
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6)
            
            frame_size = self.width * self.height * 3
            
            # Reset delay on successful start? Not yet, only after frames.
            start_time = time.time()
            frames_read_this_session = 0
            
            while self.running and self.process.poll() is None:
                try:
                    # Read raw bytes
                    # Use select to allow timeout
                    import select
                    if self.process and self.process.stdout:
                        rlist, _, _ = select.select([self.process.stdout], [], [], 5.0) # 5s timeout
                        if not rlist:
                            print(f"Stream: Read Timed Out (5s). Restarting...")
                            break
                    
                    in_bytes = self.process.stdout.read(frame_size)
                    
                    byte_count = len(in_bytes)
                    if byte_count != frame_size:
                        # Incomplete read = stream end or error
                        print(f"Stream: Incomplete read ({byte_count}/{frame_size}). Restarting...")
                        
                        # Capture error details
                        stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                        if stderr_output:
                            print(f"FFmpeg Error Log:\n{stderr_output}")
                            
                        break
                        
                    # Convert to numpy array
                    frame = np.frombuffer(in_bytes, np.uint8).reshape((self.height, self.width, 3))
                    
                    with self.lock:
                        self.grabbed = True
                        self.frame = frame
                        self.frame_id += 1
                        
                    frames_read_this_session += 1
                    if frames_read_this_session > 30:
                        # Consider stable after 30 frames
                        retry_delay = 2 
                        
                except Exception as e:
                    print(f"Stream Error: {e}")
                    break
            
            # Cleanup and retry
            self._close_process()
            if self.running:
                print(f"Stream: Connection lost. Restarting loop in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60) # Cap at 60s max backup

class StreamManager:
    def __init__(self):
        self.clients = []
        self.logger = logging.getLogger(__name__)

    def subscribe(self):
        """
        Registers a new client and yields messages from their queue.
        This generator function keeps the connection open.
        """
        q = queue.Queue()
        self.clients.append(q)
        self.logger.info(f"Client subscribed. Total clients: {len(self.clients)}")
        
        try:
            while True:
                msg = q.get()
                yield f"data: {msg}\n\n"
        except GeneratorExit:
            self.clients.remove(q)
            self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        except Exception as e:
            self.logger.error(f"Error in stream: {e}")
            if q in self.clients:
                self.clients.remove(q)

    def publish_event(self, event):
        """
        Publishes an event to all connected clients.
        
        Args:
            event (dict): The event data (timestamp, camera_id, etc.)
        """
        msg = json.dumps({'type': 'event', 'data': event})
        self._broadcast(msg)

    def publish_stats(self, stats):
        """
        Publishes stats to all connected clients.
        
        Args:
            stats (dict): Dictionary of camera stats.
        """
        msg = json.dumps({'type': 'stats', 'data': stats})
        self._broadcast(msg)

    def _broadcast(self, msg):
        """Helper to put message in all client queues."""
        for q in self.clients:
            q.put(msg)
