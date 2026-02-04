import json
import os
import time
import logging
from typing import Dict, List, Optional
from camera import VideoCamera

CONFIG_FILE = 'config.json'

class CameraManager:
    """
    Manages multiple VideoCamera instances.
    """
    def __init__(self):
        self.cameras: Dict[str, VideoCamera] = {}
        self.config = {}
        self.reload_config()

    def reload_config(self):
        """Loads config and initializes cameras."""
        if not os.path.exists(CONFIG_FILE):
             # Default config if none exists
            self.config = {
                "cameras": [],
                "snapshot_dir": "static/snapshots",
                "log_cooldown": 60
            }
            self.save_config()
        else:
            with open(CONFIG_FILE, 'r') as f:
                self.config = json.load(f)

        # Initialize cameras from config
        # Check for legacy config format and migrate if needed (simple check)
        if 'video_source' in self.config:
            self._migrate_legacy_config()

        # Stop removed cameras
        current_ids = set(cam['id'] for cam in self.config.get('cameras', []))
        active_ids = set(self.cameras.keys())
        
        for cam_id in active_ids - current_ids:
            print(f"Stopping removed camera: {cam_id}")
            self.remove_camera(cam_id)

        # Start/Update cameras
        for cam_config in self.config.get('cameras', []):
            cam_id = cam_config['id']
            # Inject global settings if missing
            if 'snapshot_dir' not in cam_config:
                cam_config['snapshot_dir'] = self.config.get('snapshot_dir', 'static/snapshots')
            if 'log_cooldown' not in cam_config:
                cam_config['log_cooldown'] = self.config.get('log_cooldown', 60)
            
            if cam_id in self.cameras:
                # Check if we need to restart (simple: restart if config changed? 
                # For now, let's just restart if requested, otherwise assume running is fine.
                # Actually, if we are reloading config, we probably want to apply changes.
                # Comparing configs is hard. Let's restart.
                print(f"Restarting camera: {cam_id}")
                self.remove_camera(cam_id)
                self.add_camera(cam_config)
            else:
                print(f"Starting new camera: {cam_id}")
                self.add_camera(cam_config)

    def _migrate_legacy_config(self):
        print("Migrating legacy configuration config.json...")
        legacy_config = self.config.copy()
        
        # Create a primary camera entry
        primary_cam = {
            "id": "primary",
            "name": "Primary Camera",
            "video_source": legacy_config.get('video_source'),
            "resolution": legacy_config.get('resolution', [0, 0]),
            "max_fps": legacy_config.get('max_fps', 10),
            "face_match_tolerance": legacy_config.get('face_match_tolerance', 0.50),
            "motion_threshold": legacy_config.get('motion_threshold', 10000),
            "webhook_url": legacy_config.get('webhook_url', '')
        }
        
        self.config = {
            "cameras": [primary_cam],
            "snapshot_dir": legacy_config.get('snapshot_dir', 'static/snapshots'),
            "log_cooldown": legacy_config.get('log_cooldown', 60)
        }
        self.save_config()

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def add_camera(self, config: Dict):
        cam_id = config.get('id')
        if not cam_id:
            return # Invalid config
            
        try:
            # We need to update VideoCamera to accept ID/Name, 
            # but for now we pass it in config which is passed to __init__
            cam = VideoCamera(config)
            self.cameras[cam_id] = cam
        except Exception as e:
            print(f"Failed to start camera {cam_id}: {e}")

    def remove_camera(self, cam_id: str):
        if cam_id in self.cameras:
            cam = self.cameras[cam_id]
            cam.running = False
            if cam.stream:
                cam.stream.stop()
            del self.cameras[cam_id]

    def get_camera(self, cam_id: str) -> Optional[VideoCamera]:
        return self.cameras.get(cam_id)

    def get_all_cameras(self) -> List[Dict]:
        """Returns list of camera metadata (id, name, status)."""
        info = []
        for cam_id, cam in self.cameras.items():
            info.append({
                "id": cam_id,
                "name": cam.config.get('name', cam_id),
                "status": "ONLINE" if cam.running else "OFFLINE",
                "stats": cam.get_stats()
            })
        return info
