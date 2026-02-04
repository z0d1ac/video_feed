
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.getcwd())

# Mock modules needed for app import
sys.modules['cv2'] = MagicMock()
# We need real auth logic so we might need real flask_login
# app imports database, so ensure it uses test db
import database
database.DB_NAME = 'test_app_db.db'

# Patch CameraManager to avoid hardware interaction
with patch('camera_manager.CameraManager') as MockManager:
    from app import app, User

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SECRET_KEY'] = 'test'
        self.client = app.test_client()
        self.test_db = 'test_app_db.db'
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        database.init_db()
        
    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def login(self):
        # The app uses a simple check: password == 'admin' (default)
        return self.client.post('/login', data={'password': 'admin'}, follow_redirects=True)

    def test_login_required(self):
        """Verify routes are protected."""
        response = self.client.get('/')
        self.assertNotEqual(response.status_code, 200)
        # Should redirect to login
        self.assertEqual(response.status_code, 302)
        
    def test_successful_login(self):
        response = self.login()
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Video Feed', response.data)
        
    def test_health_check(self):
        """Health check should be public."""
        with patch('app.get_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.config = {'snapshot_dir': '.'}
            mock_manager.get_all_cameras.return_value = []
            mock_get_manager.return_value = mock_manager
            
            response = self.client.get('/health')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'OK: All systems operational', response.data)

    def test_stream_endpoint(self):
        """Test that /api/stream allows connection and returns event-stream."""
        # Use a mock for get_stream_manager
        with patch('app.get_stream_manager') as mock_gsm:
            mock_sm = MagicMock()
            # Mock subscribe to return an empty generator or one item
            def mock_gen():
                yield "data: test\n\n"
            mock_sm.subscribe.return_value = mock_gen()
            
            mock_gsm.return_value = mock_sm
            
            # Login first
            self.login()
            
            response = self.client.get('/api/stream')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'text/event-stream')
            
    def test_stats_api(self):
        """Verify explicit stats call works (legacy/init)."""
        self.login()
        with patch('app.get_manager') as mock_mgr:
            mock_mgr.return_value.get_all_cameras.return_value = []
            response = self.client.get('/api/stats')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'application/json')

if __name__ == '__main__':
    unittest.main()
