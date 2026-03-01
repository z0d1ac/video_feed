import sqlite3
import json
import time
import os
from datetime import datetime
from zoneinfo import ZoneInfo
DB_NAME = 'video_feed.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Table for known faces
    c.execute('''
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL,
            created_at REAL,
            snapshot_path TEXT
        )
    ''')
    
    # Migration: Check if snapshot_path exists in known_faces, add if not
    c.execute("PRAGMA table_info(known_faces)")
    columns = [info[1] for info in c.fetchall()]
    if 'snapshot_path' not in columns:
        print("Migrating DB: Adding snapshot_path to known_faces")
        c.execute("ALTER TABLE known_faces ADD COLUMN snapshot_path TEXT")

    # Table for events (motion, detection)
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            event_type TEXT,
            snapshot_path TEXT,
            camera_id TEXT
        )
    ''')
    
    # Migration: Check if camera_id exists in events
    c.execute("PRAGMA table_info(events)")
    columns = [info[1] for info in c.fetchall()]
    if 'camera_id' not in columns:
        print("Migrating DB: Adding camera_id to events")
        c.execute("ALTER TABLE events ADD COLUMN camera_id TEXT")
    
    # Table for unknown faces to be reviewed
    c.execute('''
        CREATE TABLE IF NOT EXISTS unknown_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            encoding TEXT NOT NULL,
            snapshot_path TEXT,
            is_reviewed INTEGER DEFAULT 0,
            camera_id TEXT
        )
    ''')
    
    
    # Migration: Check if camera_id exists in unknown_faces
    c.execute("PRAGMA table_info(unknown_faces)")
    columns = [info[1] for info in c.fetchall()]
    if 'camera_id' not in columns:
        print("Migrating DB: Adding camera_id to unknown_faces")
        c.execute("ALTER TABLE unknown_faces ADD COLUMN camera_id TEXT")
        
    # Migration: Add detection_score to unknown_faces
    if 'detection_score' not in columns:
        print("Migrating DB: Adding detection_score to unknown_faces")
        c.execute("ALTER TABLE unknown_faces ADD COLUMN detection_score REAL")
        
    # Migration: Add predicted_name to unknown_faces
    if 'predicted_name' not in columns:
        print("Migrating DB: Adding predicted_name to unknown_faces")
        c.execute("ALTER TABLE unknown_faces ADD COLUMN predicted_name TEXT")
    
    # Migration: Add encoding_version to known_faces (v1=dlib 128-d, v2=ArcFace 512-d)
    c.execute("PRAGMA table_info(known_faces)")
    kf_columns = [info[1] for info in c.fetchall()]
    if 'encoding_version' not in kf_columns:
        print("Migrating DB: Adding encoding_version to known_faces (existing rows → v1)")
        c.execute("ALTER TABLE known_faces ADD COLUMN encoding_version INTEGER DEFAULT 1")
    
    # Migration: Add encoding_version to unknown_faces
    if 'encoding_version' not in columns:
        print("Migrating DB: Adding encoding_version to unknown_faces (existing rows → v1)")
        c.execute("ALTER TABLE unknown_faces ADD COLUMN encoding_version INTEGER DEFAULT 1")
    
    # Migration: Fix known_faces rows with NULL encoding_version (bug in resolve_unknown_face)
    c.execute("SELECT COUNT(*) FROM known_faces WHERE encoding_version IS NULL")
    null_count = c.fetchone()[0]
    if null_count > 0:
        print(f"Migrating DB: Fixing {null_count} known_faces rows with NULL encoding_version")
        # Detect version by encoding length: 512-d = ArcFace (v2), 128-d = dlib (v1)
        c.execute("SELECT id, encoding FROM known_faces WHERE encoding_version IS NULL")
        for row in c.fetchall():
            enc = json.loads(row['encoding'])
            version = 2 if len(enc) == 512 else 1
            c.execute("UPDATE known_faces SET encoding_version = ? WHERE id = ?", (version, row['id']))
        print(f"Fixed encoding_version for {null_count} rows")
    
    conn.commit()
    conn.close()

def add_known_face(name, encoding, snapshot_path=None):
    conn = get_db_connection()
    c = conn.cursor()
    # encoding is a numpy array, convert to list then json string
    encoding_json = json.dumps(encoding.tolist())
    c.execute('INSERT INTO known_faces (name, encoding, created_at, snapshot_path, encoding_version) VALUES (?, ?, ?, ?, 2)',
              (name, encoding_json, time.time(), snapshot_path))
    conn.commit()
    conn.close()

def get_known_faces(sort_by='name', version=2):
    """Loads known faces. version=2 for ArcFace, version=None for all."""
    conn = get_db_connection()
    c = conn.cursor()
    
    version_filter = 'WHERE encoding_version = ?' if version else ''
    params = (version,) if version else ()
    
    if sort_by == 'date':
        c.execute(f'SELECT id, name, encoding, created_at, snapshot_path FROM known_faces {version_filter} ORDER BY created_at DESC', params)
    else:
        # Default to name
        c.execute(f'SELECT id, name, encoding, created_at, snapshot_path FROM known_faces {version_filter} ORDER BY name ASC', params)
        
    rows = c.fetchall()
    faces = []
    for row in rows:
        faces.append({
            'id': row['id'],
            'name': row['name'],
            'encoding': json.loads(row['encoding']),
            'created_at': row['created_at'],
            'snapshot_path': row['snapshot_path']
        })
    conn.close()
    return faces

def delete_known_face(id):
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get file path first
    c.execute('SELECT snapshot_path FROM known_faces WHERE id = ?', (id,))
    row = c.fetchone()
    if row and row['snapshot_path']:
        if os.path.exists(row['snapshot_path']):
            try:
                os.remove(row['snapshot_path'])
            except OSError as e:
                print(f"Error deleting file {row['snapshot_path']}: {e}")
                
    c.execute('DELETE FROM known_faces WHERE id = ?', (id,))
    conn.commit()
    conn.close()

def update_known_face_name(id, name):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('UPDATE known_faces SET name = ? WHERE id = ?', (name, id))
    conn.commit()
    conn.close()


# Event Listeners
EVENT_LISTENERS = []

def register_event_listener(listener):
    """Registers a function to be called when a new event is logged."""
    EVENT_LISTENERS.append(listener)

def log_event(event_type, snapshot_path, camera_id="primary"):
    conn = get_db_connection()
    c = conn.cursor()
    
    timestamp = time.time()
    
    c.execute('INSERT INTO events (timestamp, event_type, snapshot_path, camera_id) VALUES (?, ?, ?, ?)',
              (timestamp, event_type, snapshot_path, camera_id))
    
    conn.commit()
    conn.close()
    
    # Notify listeners
    tz = ZoneInfo("Europe/Paris")
    event_data = {
        'timestamp': datetime.fromtimestamp(timestamp, tz).strftime('%Y-%m-%d %H:%M:%S'),
        'event_type': event_type,
        'camera_id': camera_id,
        'snapshot_path': snapshot_path
    }
    
    for listener in EVENT_LISTENERS:
        try:
            listener(event_data)
        except Exception as e:
            print(f"Error in event listener: {e}")


def add_unknown_face(encoding, snapshot_path, camera_id="primary", detection_score=0.0, predicted_name=None):
    conn = get_db_connection()
    c = conn.cursor()
    encoding_json = json.dumps(encoding.tolist())
    c.execute('INSERT INTO unknown_faces (timestamp, encoding, snapshot_path, camera_id, detection_score, predicted_name, encoding_version) VALUES (?, ?, ?, ?, ?, ?, 2)',
              (time.time(), encoding_json, snapshot_path, camera_id, detection_score, predicted_name))
    conn.commit()
    conn.close()

def get_unreviewed_unknown_faces():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM unknown_faces WHERE is_reviewed = 0 ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return rows

def resolve_unknown_face(id, name=None):
    """
    If name is provided, add to known_faces. ALWAYS mark as reviewed.
    Also retroactively updates the event log for this detection.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get details of the unknown face before resolving
    c.execute('SELECT encoding, snapshot_path, timestamp, camera_id FROM unknown_faces WHERE id = ?', (id,))
    row = c.fetchone()
    
    if row:
        timestamp = row['timestamp']
        camera_id = row['camera_id'] if 'camera_id' in row.keys() else None
        
        # Mark as reviewed
        c.execute('UPDATE unknown_faces SET is_reviewed = 1 WHERE id = ?', (id,))
        
        if name:
            encoding_json = row['encoding']
            snapshot_path = row['snapshot_path']
            
            # 1. Add to known_faces
            c.execute('INSERT INTO known_faces (name, encoding, created_at, snapshot_path, encoding_version) VALUES (?, ?, ?, ?, 2)',
                      (name, encoding_json, time.time(), snapshot_path))
            
            # 2. Retroactively update Events Log
            # Find events starting with "Unknown Person" within +/- 2 seconds of the snapshot
            # and update them to "Known Person: <Name>"
            time_window = 2.0 
            start_time = timestamp - time_window
            end_time = timestamp + time_window
            
            new_event_title = f"Known Person: {name} (Retroactive)"
            
            print(f"Retro-updating events between {start_time} and {end_time}...")
            
            # Use LIKE to match "Unknown Person..." formatting variations
            query = '''
                UPDATE events 
                SET event_type = ? 
                WHERE event_type LIKE 'Unknown Person%' 
                AND timestamp BETWEEN ? AND ?
            '''
            params = [new_event_title, start_time, end_time]
            
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            c.execute(query, tuple(params))
            
            if c.rowcount > 0:
                print(f"Updated {c.rowcount} event logs to '{new_event_title}'")
        else:
            # Ignore action - Delete the file to save space
            snapshot_path = row['snapshot_path']
            if snapshot_path and os.path.exists(snapshot_path):
                try:
                    os.remove(snapshot_path)
                    print(f"Deleted ignored snapshot: {snapshot_path}")
                except OSError as e:
                    print(f"Error deleting ignored file {snapshot_path}: {e}")
                
                # Verify we don't wipe the path from DB if we want to keep history? 
                # Actually, strictly speaking the row still exists. 
                # We can optionally clear the path column to indicate no file exists.
                c.execute('UPDATE unknown_faces SET snapshot_path = NULL WHERE id = ?', (id,))

    conn.commit()
    conn.close()

def get_recent_events(limit=10):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

if __name__ == '__main__':
    init_db()
    print("Database initialized.")
