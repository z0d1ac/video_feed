import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestStartupLogic(unittest.TestCase):
    

    def test_production_mode_init(self):
        """Test that services init immediately when DEBUG is False."""
        if 'app' in sys.modules: del sys.modules['app']
        # We need to mock heavy dependencies to avoid missing module errors in test envs
        # and to avoid starting real threads during tests if possible.
        # But for this test, we accept the threads starting or mock them.
        # Let's mock the heavy systems to speed it up.
        
        with patch.dict(os.environ, {'DEBUG': 'False', 'WERKZEUG_RUN_MAIN': 'false'}):
            # Patch CameraManager to avoid real hardware access
            with patch('camera_manager.CameraManager') as MockCamMan:
                import app
                # Check if get_manager was called effectively (by checking if global is set)
                # But get_manager sets a local global `camera_manager`.
                # Since we mock the CLASS, `app.camera_manager` will be the mock instance.
                self.assertIsNotNone(app.camera_manager)
                print("Production Mode: SUCCESS")

    def test_debug_parent_process_skip(self):
        """Test that services DO NOT init in the reloader parent process."""
        if 'app' in sys.modules: del sys.modules['app']
        
        with patch.dict(os.environ, {'DEBUG': 'True', 'WERKZEUG_RUN_MAIN': 'false'}):
            with patch('camera_manager.CameraManager') as MockCamMan:
                import app
                self.assertIsNone(app.camera_manager)
                print("Debug Parent: SUCCESS (Skipped)")

    def test_debug_child_process_init(self):
        """Test that services init in the reloader child process."""
        if 'app' in sys.modules: del sys.modules['app']
        
        with patch.dict(os.environ, {'DEBUG': 'True', 'WERKZEUG_RUN_MAIN': 'true'}):
            with patch('camera_manager.CameraManager') as MockCamMan:
                import app
                self.assertIsNotNone(app.camera_manager)
                print("Debug Child: SUCCESS")

if __name__ == '__main__':
    unittest.main()
