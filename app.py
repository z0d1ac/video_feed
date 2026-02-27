from flask import Flask, render_template, Response, request, redirect, url_for, flash, abort
from functools import wraps
import database
from camera_manager import CameraManager
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import json
import shutil
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from stream import StreamManager
from dotenv import load_dotenv
from auth_models import User
from flasgger import Swagger

# Load Env
load_dotenv()

app = Flask(__name__)
# Secure Secret Key
app.secret_key = os.getenv('SECRET_KEY', 'default-dev-key-please-change')

from functools import wraps
from flask import Flask, render_template, Response, request, redirect, url_for, flash, abort

# ... imports ...

# Swagger config
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        },
        {
             "endpoint": 'openapi',
             "route": '/openapi.json',
             "rule_filter": lambda rule: True,
             "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs"
}

template = {
    "swagger": "2.0",
    "info": {
        "title": "Video Feed API",
        "description": "API for accessing cameras, snapshots, and events.",
        "version": "1.0.0"
    },
    "securityDefinitions": {
        "BasicAuth": {
            "type": "basic"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "name": "X-API-Key",
            "in": "header"
        }
    },
    "security": [
        {"ApiKeyAuth": []},
        {"BasicAuth": []}
    ]
}

swagger = Swagger(app, config=swagger_config, template=template)

# Auth Helper
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Allow Session Auth (Browser/UI)
        if current_user.is_authenticated:
            return f(*args, **kwargs)
        
        # 2. Check API Key
        # Get key from config
        manager = get_manager()
        server_key = manager.config.get('api_key')
        
        if not server_key:
            # If no key configured, fallback to strict login required (or deny)
            # Default to denying if no key is set to avoid accidental open API
            return "API Key not configured", 401
            
        # Check Header or Query Param
        request_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if request_key and request_key == server_key:
            return f(*args, **kwargs)
            
        return "Unauthorized: Invalid API Key", 401
    return decorated_function

# Auth Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple Admin User Loader
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Global camera manager instance
camera_manager = None
# Global stream manager
stream_manager = None

def get_manager():
    global camera_manager
    if camera_manager is None:
        camera_manager = CameraManager()
    return camera_manager

def get_stream_manager():
    global stream_manager
    if stream_manager is None:
        stream_manager = StreamManager()
        # Register event listener
        database.register_event_listener(stream_manager.publish_event)
        # Start stats thread
        threading.Thread(target=stats_monitor, daemon=True).start()
    return stream_manager

def stats_monitor():
    """Background thread to publish stats periodically."""
    last_stats = {}
    while True:
        try:
            mgr = get_manager()
            cameras = mgr.get_all_cameras()
            stats = {}
            for cam in cameras:
                stats[cam['id']] = cam['stats']
            
            # Publish stats (could curb to only when changed, but FPS always changes)
            # For now, 1 second interval
            s_mgr = get_stream_manager()
            
            # Add system stats
            tz = ZoneInfo("Europe/Paris")
            stats['system'] = {
                'time': datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
            }

            s_mgr.publish_stats(stats)
            
        except Exception as e:
            print(f"Stats monitor error: {e}")
            
        time.sleep(1)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        password = request.form.get('password')
        admin_pass = os.getenv('ADMIN_PASSWORD', 'admin')
        
        if password == admin_pass:
            user = User('admin')
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Access Denied: Invalid Credentials', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    events = database.get_recent_events(limit=10)
    # Format events
    formatted_events = []
    tz = ZoneInfo("Europe/Paris")
    for event in events:
        # timestamp is unix float
        dt = datetime.fromtimestamp(event['timestamp'], tz)
        formatted_events.append({
            'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'event_type': event['event_type'],
            'camera_id': event['camera_id'] if 'camera_id' in event.keys() else 'Unknown',
            'snapshot_path': event['snapshot_path']
        })
    
    cameras = get_manager().get_all_cameras()
    return render_template('index.html', events=formatted_events, cameras=cameras)

def gen(camera):
    while True:
        frame = camera.get_frame()
        
        if frame is None:
            # Feed is stalled
            try:
                with open('static/no_signal.png', 'rb') as f:
                    frame = f.read()
            except:
                # Fallback if image missing
                pass
        
        if frame:
            yield (b'--boundarydonotcross\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Limit streaming to ~20 FPS to save CPU
        # OpenCV encoding + Motion Detection every frame is expensive
        time.sleep(0.05)

@app.route('/api/stream')
@require_api_key
def api_stream():
    """Server-Sent Events endpoint."""
    return Response(get_stream_manager().subscribe(), mimetype="text/event-stream")

@app.route('/api/stats')
@require_api_key
def api_stats():
    """
    Get system and camera statistics.
    ---
    tags:
      - System
    operationId: getSystemStats
    responses:
      200:
        description: System statistics including camera FPS and server time.
    """
    cameras = get_manager().get_all_cameras()
    stats = {}
    for cam in cameras:
        stats[cam['id']] = cam['stats']
        
    tz = ZoneInfo("Europe/Paris")
    stats['system'] = {
        'time': datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    }
    return json.dumps(stats), 200, {'Content-Type': 'application/json'}

@app.route('/api/cameras')
@require_api_key
def api_cameras():
    """
    List all configured cameras.
    ---
    tags:
      - Cameras
    operationId: getCameras
    responses:
      200:
        description: List of configured cameras with their status and settings.
    """
    cameras = get_manager().get_all_cameras()
    return json.dumps(cameras), 200, {'Content-Type': 'application/json'}

@app.route('/api/snapshot/<string:camera_id>')
@require_api_key
def api_snapshot(camera_id):
    """
    Get current snapshot from a camera.
    ---
    tags:
      - Cameras
    operationId: getSnapshot
    parameters:
      - name: camera_id
        in: path
        type: string
        required: true
        description: ID of the camera
    responses:
      200:
        description: JPEG image of the current frame
        content:
          image/jpeg:
            schema:
              type: string
              format: binary
      404:
        description: Camera not found
    """
    manager = get_manager()
    camera = manager.get_camera(camera_id)
    if not camera:
        return "Camera not found", 404
        
    frame = camera.get_frame()
    if frame is None:
        try:
            with open('static/no_signal.png', 'rb') as f:
                frame = f.read()
        except:
             return "No signal and missing placeholder", 404

    return Response(frame, mimetype='image/jpeg')

@app.route('/api/events')
@require_api_key
def api_events():
    """
    Get recent detection events.
    ---
    tags:
      - Events
    operationId: getEvents
    responses:
      200:
        description: List of 10 most recent events.
    """
    events = database.get_recent_events(limit=10)
    formatted_events = []
    tz = ZoneInfo("Europe/Paris")
    for event in events:
        dt = datetime.fromtimestamp(event['timestamp'], tz)
        formatted_events.append({
            'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'), # Full date time in CET
            'event_type': event['event_type'],
            'camera_id': event['camera_id'] if 'camera_id' in event.keys() else 'Unknown',
            'snapshot_path': event['snapshot_path']
        })
    return json.dumps(formatted_events), 200, {'Content-Type': 'application/json'}

@app.route('/api/faces', methods=['GET'])
@require_api_key
def api_get_faces():
    """
    List all known faces.
    ---
    tags:
      - Faces
    operationId: getKnownFaces
    responses:
      200:
        description: List of known faces.
    """
    faces = database.get_known_faces(version=None)
    # Remove large encoding data for list view
    for face in faces:
        if 'encoding' in face:
            del face['encoding']
    return json.dumps(faces), 200, {'Content-Type': 'application/json'}

@app.route('/api/faces/<int:id>', methods=['DELETE'])
@require_api_key
def api_delete_face(id):
    """
    Delete a known face.
    ---
    tags:
      - Faces
    operationId: deleteKnownFace
    parameters:
      - name: id
        in: path
        type: integer
        required: true
        description: ID of the face to delete
    responses:
      200:
        description: Face deleted successfully
    """
    database.delete_known_face(id)
    # Reload FR system
    manager = get_manager()
    for cam in manager.cameras.values():
        cam.fr_system.reload_known_faces()
        
    return json.dumps({'success': True, 'message': 'Face deleted'}), 200, {'Content-Type': 'application/json'}

@app.route('/api/faces/unknown', methods=['GET'])
@require_api_key
def api_get_unknown_faces():
    """
    List unreviewed unknown faces.
    ---
    tags:
      - Faces
    operationId: getUnknownFaces
    responses:
      200:
        description: List of unreviewed unknown faces.
    """
    unknown_faces = database.get_unreviewed_unknown_faces()
    faces = []
    for face in unknown_faces:
         # Convert row to dict
        faces.append({
            'id': face['id'],
            'timestamp': face['timestamp'],
            'snapshot_path': face['snapshot_path'],
            'camera_id': face['camera_id'] if 'camera_id' in face.keys() else 'unknown',
            'detection_score': face['detection_score'] if 'detection_score' in face.keys() else None,
            'predicted_name': face['predicted_name'] if 'predicted_name' in face.keys() else None
        })
    return json.dumps(faces), 200, {'Content-Type': 'application/json'}

@app.route('/api/faces/<int:id>/tag', methods=['POST'])
@require_api_key
def api_tag_face(id):
    """
    Tag or ignore an unknown face.
    ---
    tags:
      - Faces
    operationId: tagUnknownFace
    parameters:
      - name: id
        in: path
        type: integer
        required: true
        description: ID of the unknown face
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
              description: Name to tag the face with. If null, face is ignored? No, use action.
            action:
              type: string
              enum: [save, ignore]
              description: Action to take.
    responses:
      200:
        description: Face tagged successfully
      400:
        description: Invalid input
    """
    data = request.get_json()
    if not data:
        return json.dumps({'error': 'Missing JSON body'}), 400, {'Content-Type': 'application/json'}
        
    action = data.get('action')
    name = data.get('name')
    
    if action == 'ignore':
        database.resolve_unknown_face(id, name=None)
        return json.dumps({'success': True, 'message': 'Face ignored'}), 200, {'Content-Type': 'application/json'}
    elif action == 'save':
        if not name:
            return json.dumps({'error': 'Name is required for save action'}), 400, {'Content-Type': 'application/json'}
            
        database.resolve_unknown_face(id, name=name)
        # Reload known faces
        manager = get_manager()
        for cam in manager.cameras.values():
            cam.fr_system.reload_known_faces()
            
        return json.dumps({'success': True, 'message': f'Face tagged as {name}'}), 200, {'Content-Type': 'application/json'}
    else:
        return json.dumps({'error': 'Invalid action'}), 400, {'Content-Type': 'application/json'}

@app.route('/video_feed/<string:camera_id>')
@login_required
def video_feed(camera_id):
    manager = get_manager()
    camera = manager.get_camera(camera_id)
    if not camera:
        return "Camera not found", 404
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=boundarydonotcross')

@app.route('/review')
@login_required
def review():
    unknown_faces = database.get_unreviewed_unknown_faces()
    # Format for template
    faces = []
    for face in unknown_faces:
        faces.append({
            'id': face['id'],
            'timestamp': face['timestamp'],
            'snapshot_path': face['snapshot_path'],
            'camera_id': face['camera_id'] if 'camera_id' in face.keys() else 'unknown',
            'detection_score': face['detection_score'] if 'detection_score' in face.keys() else None,
            'predicted_name': face['predicted_name'] if 'predicted_name' in face.keys() else None
        })
    return render_template('review.html', faces=faces)

@app.route('/tag/<int:id>', methods=['POST'])
@login_required
def tag_face(id):
    name = request.form.get('name')
    action = request.form.get('action') # 'save' or 'ignore'
    
    if action == 'ignore':
        database.resolve_unknown_face(id, name=None)
        flash('Face ignored.', 'info')
    elif name:
        database.resolve_unknown_face(id, name=name)
        # Reload known faces in ALL cameras
        manager = get_manager()
        for cam in manager.cameras.values():
            cam.fr_system.reload_known_faces()
        flash(f'Face tagged as {name}.', 'success')
    else:
        flash('Name required to save.', 'error')
        
    return redirect(url_for('review'))

@app.route('/known_faces')
@login_required
def known_faces():
    sort_by = request.args.get('sort_by', 'name')
    faces = database.get_known_faces(sort_by=sort_by, version=None)
    # Convert timestamp to readable date if present
    tz = ZoneInfo("Europe/Paris")
    for face in faces:
        if face.get('created_at'):
            dt = datetime.fromtimestamp(face['created_at'], tz)
            face['created_at'] = dt.strftime('%Y-%m-%d %H:%M:%S')
    return render_template('known_faces.html', faces=faces, active_sort=sort_by)

@app.route('/known_faces/update/<int:id>', methods=['POST'])
@login_required
def update_face(id):
    name = request.form.get('name')
    if name:
        database.update_known_face_name(id, name)
        # Reload all
        manager = get_manager()
        for cam in manager.cameras.values():
            cam.fr_system.reload_known_faces()
        flash(f'Updated name to {name}.', 'success')
    return redirect(url_for('known_faces'))

@app.route('/known_faces/delete/<int:id>', methods=['POST'])
@login_required
def delete_face(id):
    database.delete_known_face(id)
    manager = get_manager()
    for cam in manager.cameras.values():
        cam.fr_system.reload_known_faces()
    flash('Face deleted.', 'success')
    return redirect(url_for('known_faces'))

@app.route('/settings', methods=['GET'])
@login_required
def settings():
    manager = get_manager()
    return render_template('settings.html', config=manager.config)

@app.route('/settings/global', methods=['POST'])
@login_required
def update_global_settings():
    manager = get_manager()
    manager.config['snapshot_dir'] = request.form.get('snapshot_dir', 'static/snapshots')
    manager.config['log_cooldown'] = int(request.form.get('log_cooldown', 60))
    manager.save_config()
    manager.reload_config()
    flash('Global settings saved.', 'success')
    return redirect(url_for('settings'))

@app.route('/settings/camera/add', methods=['POST'])
@login_required
def add_camera():
    manager = get_manager()
    cam_id = request.form.get('id').strip().lower().replace(' ', '_')
    name = request.form.get('name')
    url = request.form.get('video_source')
    
    if not cam_id or not name or not url:
        flash('ID, Name and Video Source are required.', 'error')
        return redirect(url_for('settings'))
    
    # Check if ID exists
    for cam in manager.config['cameras']:
        if cam['id'] == cam_id:
            flash(f'Camera ID {cam_id} already exists.', 'error')
            return redirect(url_for('settings'))

    new_cam = {
        "id": cam_id,
        "name": name,
        "video_source": url,
        "resolution": [0,0],
        "max_fps": int(request.form.get('max_fps', 10)),
        "face_match_tolerance": float(request.form.get('face_match_tolerance', 0.40)),
        "min_face_score": float(request.form.get('min_face_score', 0.5)),
        "motion_threshold": int(request.form.get('motion_threshold', 10000)),
        "process_interval": int(request.form.get('process_interval', 5)),
        "webhook_url": request.form.get('webhook_url', ''),
        "process_resolution": int(request.form.get('process_resolution', 400)),
        "display_resolution": int(request.form.get('display_resolution', 800)),
        "detection_scale": float(request.form.get('detection_scale', 0.5))
    }
    
    manager.config['cameras'].append(new_cam)
    manager.save_config()
    manager.reload_config()
    flash(f'Camera {name} added.', 'success')
    return redirect(url_for('settings'))

@app.route('/settings/camera/update/<string:cam_id>', methods=['POST'])
@login_required
def update_camera(cam_id):
    manager = get_manager()
    # Find camera config
    cam_config = None
    for cam in manager.config['cameras']:
        if cam['id'] == cam_id:
            cam_config = cam
            break
    
    if not cam_config:
        flash('Camera not found.', 'error')
        return redirect(url_for('settings'))

    cam_config['name'] = request.form.get('name')
    cam_config['video_source'] = request.form.get('video_source')
    cam_config['max_fps'] = int(request.form.get('max_fps', 10))
    cam_config['face_match_tolerance'] = float(request.form.get('face_match_tolerance', 0.50))
    cam_config['min_face_score'] = float(request.form.get('min_face_score', 0.0))
    cam_config['motion_threshold'] = int(request.form.get('motion_threshold', 10000))
    cam_config['process_interval'] = int(request.form.get('process_interval', 5))
    cam_config['webhook_url'] = request.form.get('webhook_url', '')
    cam_config['process_resolution'] = int(request.form.get('process_resolution', 400))
    cam_config['display_resolution'] = int(request.form.get('display_resolution', 800))
    cam_config['detection_scale'] = float(request.form.get('detection_scale', 0.5))
    
    # Reset resolution if re-probing requested (maybe a checkbox? or just if URL changed?)
    # For now, let's just always reset to [0,0] if they save settings, 
    # ensuring the system adapts if they changed the stream URL.
    # Ideally only if URL changed, but simple is better for now.
    cam_config['resolution'] = [0, 0]

    manager.save_config()
    manager.reload_config()
    flash(f'Camera {cam_config["name"]} updated.', 'success')
    return redirect(url_for('settings'))

@app.route('/settings/camera/delete/<string:cam_id>', methods=['POST'])
@login_required
def delete_camera(cam_id):
    manager = get_manager()
    manager.config['cameras'] = [c for c in manager.config['cameras'] if c['id'] != cam_id]
    manager.save_config()
    manager.reload_config()
    flash('Camera deleted.', 'success')
    return redirect(url_for('settings'))

@app.route('/api/test_rtsp', methods=['POST'])
@login_required
def test_rtsp():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return json.dumps({'success': False, 'message': 'No URL provided'}), 400, {'Content-Type': 'application/json'}
        
    try:
        # Simple FFprobe check (timeout 5s)
        cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 -rtsp_transport tcp '{url}'"
        print(f"Testing RTSP: {cmd}")
        
        # subprocess.check_output will raise CalledProcessError if it fails (non-zero exit)
        import subprocess
        subprocess.check_output(cmd, shell=True, timeout=8)
        
        return json.dumps({'success': True, 'message': 'Connection Successful!'}), 200, {'Content-Type': 'application/json'}
        
    except subprocess.TimeoutExpired:
        return json.dumps({'success': False, 'message': 'Connection Timed Out'}), 200, {'Content-Type': 'application/json'}
    except subprocess.CalledProcessError:
        return json.dumps({'success': False, 'message': 'Connection Failed (Invalid Stream)'}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return json.dumps({'success': False, 'message': f'Error: {str(e)}'}), 200, {'Content-Type': 'application/json'}



@app.route('/health')
def health_check():
    """
    Health check endpoint for Nagios/Monitoring.
    ---
    tags:
      - System
    operationId: getHealth
    responses:
      200:
        description: System healthy
      500:
        description: System critical error (DB, Storage, or Cameras)
    """
    
    # 1. Database Check
    try:
        conn = database.get_db_connection()
        # Verify table exists to ensure DB is initialized
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events';")
        conn.close()
    except Exception as e:
        return "CRITICAL: Database connection failed or invalid", 500

    # 2. Storage Check
    manager = get_manager()
    snapshot_dir = manager.config.get('snapshot_dir', 'static/snapshots')
    
    if not os.path.exists(snapshot_dir):
        return "CRITICAL: Snapshot directory missing", 500
    
    if not os.access(snapshot_dir, os.W_OK):
        return "CRITICAL: Snapshot directory readonly", 500
        
    try:
        total, used, free = shutil.disk_usage(snapshot_dir)
        free_mb = free / (1024 * 1024)
        if free_mb < 500:
            return f"CRITICAL: Low disk space ({int(free_mb)}MB free)", 500
    except Exception as e:
        return f"CRITICAL: Failed to check disk usage: {str(e)}", 500

    # 3. Camera Check
    cameras = manager.get_all_cameras()
    if not cameras:
        # Warning if no cameras configured? Or maybe that's fine?
        # Let's say it's OK but note it.
        pass
        
    for cam in cameras:
        # Check Online Status
        if cam['status'] != 'ONLINE':
            return f"CRITICAL: Camera '{cam['name']}' is OFFLINE", 500
            
        # Check for Stalled Stream (0 FPS)
        # Note: Stats returns 'fps' as float
        fps = cam['stats'].get('fps', 0.0)
        if fps == 0.0:
            return f"CRITICAL: Camera '{cam['name']}' is stalled (0 FPS)", 500

    # Success
    # Build a nice info string
    active_cams = len(cameras)
    free_gb = round(free / (1024 * 1024 * 1024), 2)
    
    return f"OK: All systems operational | storage_free_gb={free_gb} cameras_active={active_cams}", 200

# Initialization Logic
def initialize_app_services():
    """Initializes all backend services (DB, Cameras, Streams)."""
    # Prevent double auth if called multiple times, though manager is singleton
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing App Services...")
    
    # 1. Database
    database.init_db()
    
    # 2. Camera Manager (Starts Camera Threads)
    get_manager()
    
    # 3. Stream Manager (Starts Background Stats)
    get_stream_manager()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] App Services Initialized.")

# Execute Initialization
# We want this to run:
# 1. When running via 'python app.py' (Production/Dev manual)
# 2. When running via 'flask run' (Dev)
# 3. When running via Gunicorn (Production)
#
# BUT we must avoid running it TWICE when using the Werkzeug reloader (Debug Mode).
# Werkzeug runs the script once to start the reloader (parent), and then again for the child.
# We only want the CHILD to run the heavy initialization.

debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'

# Check if we are in the reloader child process (WERKZEUG_RUN_MAIN='true')
# OR if we are not using the reloader at all (e.g. Production/Gunicorn)
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not debug_mode:
    with app.app_context():
        initialize_app_services()

if __name__ == '__main__':
    # Run app
    app.run(host='0.0.0.0', port=5050, threaded=True, debug=debug_mode)
