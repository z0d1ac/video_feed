
import unittest
import os
import sqlite3
import numpy as np
import sys

sys.path.append(os.getcwd())
import database

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.test_db = 'test_unit_db.db'
        database.DB_NAME = self.test_db
        # Ensure fresh start
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        database.init_db()
        
    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def test_add_known_face(self):
        encoding = np.array([0.1, 0.2, 0.3])
        database.add_known_face("TestU", encoding)
        
        faces = database.get_known_faces()
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]['name'], "TestU")
        # Check encoding restoration
        self.assertTrue(np.allclose(faces[0]['encoding'], [0.1, 0.2, 0.3]))
        
    def test_retroactive_update(self):
        """Test the fix for retroactive identification of events."""
        # 1. Add Unknown Face
        database.add_unknown_face(np.zeros(128), "path/to/snap.jpg", camera_id="cam1", detection_score=0.8)
        
        # Get ID
        c = database.get_db_connection()
        face_id = c.execute("SELECT id FROM unknown_faces").fetchone()['id']
        ts = c.execute("SELECT timestamp FROM unknown_faces").fetchone()['timestamp']
        c.close()

        # 2. Add Event with correct format
        database.log_event("Unknown Person (Score: 0.8) on TestCam", "path/to/snap.jpg", camera_id="cam1")
        # Override timestamp to match exactly for test simplicity (in real app they are close)
        conn = database.get_db_connection()
        conn.execute("UPDATE events SET timestamp = ? WHERE camera_id = 'cam1'", (ts,))
        conn.commit()
        conn.close()
        
        # 3. Resolve
        database.resolve_unknown_face(face_id, "Bob")
        
        # 4. Verify Event
        events = database.get_recent_events(1)
        self.assertIn("Known Person: Bob", events[0]['event_type'])

    def test_event_listener(self):
        """Test that event listeners are triggered on log_event."""
        # Define a callback
        triggered = []
        def listener(event):
            triggered.append(event)
            
        # Register
        database.register_event_listener(listener)
        
        # Log event
        database.log_event("Test Event", "snapshot.jpg")
        
        # Verify
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]['event_type'], "Test Event")
        self.assertEqual(triggered[0]['snapshot_path'], "snapshot.jpg")
        
        # Cleanup listeners for other tests? 
        # database.EVENT_LISTENERS is global. Ideally we should reset it in tearDown or use a fresh module reload.
        # For simplicity, we just clear it manually here or accept it appends.
        database.EVENT_LISTENERS.remove(listener)

if __name__ == '__main__':
    unittest.main()
