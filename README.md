# Watchtower
    
Watchtower is a robust, self-hosted security surveillance system designed for monitoring RTSP camera feeds with advanced AI capabilities.

## Key Features

- **🧠 Real-Time Face Recognition**: Automatically identifies known individuals and flags unknown visitors using OpenCV YuNet and ArcFace ONNX models.
- **⚡ Smart Throttling**: intelligently tracks unknown visitors using vector persistence to ensure distinct strangers are logged individually while preventing spam.
- **🚀 Immediate Startup**: Always-on monitoring ensures cameras are active and recording events the moment the container starts, independent of client connections.
- **🏃 Motion-Guided Detection**: Filers out static false positives (like parked bicycles or posters) by ensuring detected faces are actually moving.
- **📹 Multi-Camera Support**: Concurrent monitoring of multiple RTSP streams with individual configuration.
- **📝 Activity Logging**: Detailed event history with localized timestamps (CET) and snapshot capture.
- **🔍 Review Interface**: Built-in workflow to review, tag, or ignore unknown faces to continuously improve recognition accuracy.
- **⚡ Efficient**: Uses MOG2 background subtraction to trigger expensive AI processing only when activity is detected. Separate processing and display resolutions minimize CPU usage.
- **🐳 Docker Ready**: Zero-dependency deployment using Docker Compose. ONNX models are downloaded automatically.

## Getting Started with Docker

This application is designed to run easily within a Docker container.

### Prerequisites

- Docker
- Docker Compose

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/z0d1ac/video_feed.git
    cd video_feed
    ```

2.  **Configure the application:**
    Copy the example configuration file and edit it with your camera details.
    ```bash
    cp config.json.example config.json
    ```
    *Edit `config.json` to add your RTSP stream URLs and other settings.*
    
3.  **Configure Environment Variables:**
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    *Edit `.env` to set your `SECRET_KEY` and `ADMIN_PASSWORD`.*

4.  **Start the container:**
    Run the following command to build and start the application in detached mode:
    ```bash
    docker-compose up -d --build
    ```

5.  **Access the Application:**
    Open your web browser and navigate to:
    [http://localhost:5050](http://localhost:5050)

### Default Login
- **Username:** admin (Authentication is skipped if already logged in, but the user struct expects an ID)
- **Password:** admin (Default from `app.py`, configurable via `ADMIN_PASSWORD` env var)

### Configuration (`config.json`)
- `cameras`: List of camera objects.
  - `video_source`: RTSP URL or camera index (e.g., "0").
  - `face_match_tolerance`: Cosine distance threshold for face matching (default: `0.40`, lower = stricter).
  - `min_face_score`: YuNet face detection confidence threshold, 0.0-1.0 (default: `0.5`).
  - `motion_threshold`: Pixel count to trigger motion (default: `10000`).
  - `process_resolution`: Width in px for motion/face detection processing (default: `400`).
  - `display_resolution`: Width in px for the web stream (default: `800`).
  - `detection_scale`: Scale factor within face detection, 0.0-1.0 (default: `0.5`).
  - `process_interval`: Run face detection every N frames during motion (default: `5`).
- `snapshot_dir`: Directory to store face snapshots (mapped to volume in docker).
- `webhook_url`: Optional URL to receive JSON payloads on detection.
- `api_key`: Secure key for API authentication (e.g., "your-secret-key").

### API Documentation & Authentication

The application provides a full REST API for integrating with other systems (Home Assistant, etc.).

#### 📖 Swagger UI
Interactive API documentation is available at:
[http://localhost:5050/apidocs](http://localhost:5050/apidocs)

#### 🔐 Authentication
The API is secured using an **API Key**. 
1.  Add `"api_key": "your-secret-key"` to your `config.json`.
2.  Authenticate requests using one of the following methods:
    *   **Header**: `X-API-Key: your-secret-key`
    *   **Query Param**: `?api_key=your-secret-key`

#### Key Endpoints
*   `GET /api/cameras` - List configured cameras
*   `GET /api/snapshot/<camera_id>` - Get current frame
*   `GET /api/faces` - List known people
*   `GET /api/events` - Get recent detections

### Development
To run tests:
```bash
python3 -m pytest tests/ -v
```
