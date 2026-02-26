import cv2
import numpy as np
import database
import logging
import os
import urllib.request

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logging.basicConfig(level=logging.INFO)

# Default model paths (relative to app directory)
_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

YUNET_MODEL_PATH = os.environ.get(
    'YUNET_MODEL_PATH',
    os.path.join(_MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
)
ARCFACE_MODEL_PATH = os.environ.get(
    'ARCFACE_MODEL_PATH',
    os.path.join(_MODELS_DIR, 'w600k_r50.onnx')
)

# Download mirror URLs (tried in order; first success wins)
_MODEL_MIRRORS = {
    'face_detection_yunet_2023mar.onnx': [
        'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    ],
    'w600k_r50.onnx': [
        'https://huggingface.co/vjump21848/buffalo_l_unzip/resolve/main/w600k_r50.onnx',
        'https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx',
        'https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/w600k_r50.onnx',
    ],
}


def _ensure_models():
    """Downloads ONNX models if they are not present, trying multiple mirrors."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    
    for filename, urls in _MODEL_MIRRORS.items():
        path = os.path.join(_MODELS_DIR, filename)
        if os.path.exists(path):
            continue
        
        logging.info(f"Downloading {filename}... (this may take a moment)")
        downloaded = False
        
        for url in urls:
            try:
                logging.info(f"  Trying: {url}")
                urllib.request.urlretrieve(url, path)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logging.info(f"  Downloaded {filename} ({size_mb:.1f}MB)")
                downloaded = True
                break
            except Exception as e:
                logging.warning(f"  Mirror failed: {e}")
                # Clean up partial download
                if os.path.exists(path):
                    os.remove(path)
                continue
        
        if not downloaded:
            raise FileNotFoundError(
                f"Could not download {filename} from any mirror. "
                f"Please download manually and place at: {path}\n"
                f"Mirrors tried: {urls}"
            )


_ensure_models()

# Standard ArcFace alignment landmarks (for 112x112 target)
# These are the reference positions for left eye, right eye, nose, left mouth, right mouth
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _estimate_norm(landmarks_5):
    """Estimates similarity transform from 5 facial landmarks to ArcFace reference."""
    src = np.array(landmarks_5, dtype=np.float32)
    dst = ARCFACE_DST.copy()

    # Use cv2.estimateAffinePartial2D for a similarity transform (rotation + scale + translation)
    tform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return tform


def _align_face(img, landmarks_5, image_size=112):
    """Aligns and crops a face to 112x112 using 5 facial landmarks."""
    M = _estimate_norm(landmarks_5)
    if M is None:
        return None
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


class FacialRecognitionSystem:
    def __init__(self, tolerance: float = 0.4, min_score: float = 0.5, detection_scale: float = 1.0):
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance          # Cosine distance threshold (lower = stricter)
        self.min_score = min_score          # YuNet confidence threshold (0.0-1.0)
        self.detection_scale = detection_scale
        
        # Initialize YuNet face detector
        if not os.path.exists(YUNET_MODEL_PATH):
            raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL_PATH}")
        
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (320, 320),       # Will be updated per frame
            self.min_score,   # Score threshold
            0.3,              # NMS threshold  
            5000              # Top K
        )
        
        # Initialize ArcFace encoder via ONNX Runtime
        if not os.path.exists(ARCFACE_MODEL_PATH):
            raise FileNotFoundError(f"ArcFace model not found: {ARCFACE_MODEL_PATH}")
        
        if ort is None:
            raise ImportError("onnxruntime is required for ArcFace encoding. Install with: pip install onnxruntime")
        
        self.encoder = ort.InferenceSession(
            ARCFACE_MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        self.encoder_input_name = self.encoder.get_inputs()[0].name
        self.encoder_output_name = self.encoder.get_outputs()[0].name
        
        logging.info(f"FacialRecognitionSystem initialized (YuNet + ArcFace ONNX)")
        logging.info(f"  Tolerance: {self.tolerance}, Min Score: {self.min_score}")
        
        self.reload_known_faces()

    def reload_known_faces(self):
        """Loads known faces from the database (v2 encodings only)."""
        faces = database.get_known_faces()
        self.known_face_encodings = [np.array(face['encoding']) for face in faces]
        self.known_face_names = [face['name'] for face in faces]
        logging.info(f"Loaded {len(self.known_face_names)} known faces. Tolerance: {self.tolerance}")

    def _encode_face(self, aligned_face):
        """
        Runs ArcFace ONNX inference on a 112x112 aligned face image.
        Returns a normalized 512-d embedding vector.
        """
        # ArcFace expects (1, 3, 112, 112) float32 input, RGB, normalized to [-1, 1]
        img = aligned_face.astype(np.float32)
        # Normalize pixel values
        img = (img / 127.5) - 1.0
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        embedding = self.encoder.run(
            [self.encoder_output_name],
            {self.encoder_input_name: img}
        )[0][0]
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

    def process_frame(self, rgb_frame):
        """
        Detects faces in the frame and identifies them.
        Uses YuNet for detection and ArcFace ONNX for encoding.
        Returns a list of (top, right, bottom, left, name, encoding, distance, score) tuples.
        """
        orig_h, orig_w = rgb_frame.shape[:2]
        scale = self.detection_scale

        # 1. Prepare frame for detection
        if scale < 1.0:
            det_w = int(orig_w * scale)
            det_h = int(orig_h * scale)
            det_frame = cv2.resize(rgb_frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            det_frame = rgb_frame
            det_w, det_h = orig_w, orig_h

        # YuNet needs BGR input
        det_frame_bgr = cv2.cvtColor(det_frame, cv2.COLOR_RGB2BGR)

        # Update detector input size
        self.detector.setInputSize((det_w, det_h))

        # 2. Detect faces
        try:
            _, faces_mat = self.detector.detect(det_frame_bgr)
        except Exception as e:
            logging.error(f"YuNet detection failed: {e}")
            return []

        if faces_mat is None or len(faces_mat) == 0:
            return []

        results = []
        inv_scale = 1.0 / scale if scale < 1.0 else 1.0

        for face_row in faces_mat:
            # YuNet output: [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
            #                 nose_x, nose_y, right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y, score]
            x, y, w, h = face_row[0:4].astype(int)
            confidence = float(face_row[14])

            # Extract 5 landmarks (in detection frame coordinates)
            landmarks_det = np.array([
                [face_row[4], face_row[5]],    # right eye
                [face_row[6], face_row[7]],    # left eye
                [face_row[8], face_row[9]],    # nose
                [face_row[10], face_row[11]],  # right mouth
                [face_row[12], face_row[13]],  # left mouth
            ], dtype=np.float32)

            # Scale coordinates back to original frame
            if scale < 1.0:
                x = int(x * inv_scale)
                y = int(y * inv_scale)
                w = int(w * inv_scale)
                h = int(h * inv_scale)
                landmarks_orig = landmarks_det * inv_scale
            else:
                landmarks_orig = landmarks_det

            # Convert to (top, right, bottom, left) format
            top = max(0, y)
            left = max(0, x)
            bottom = min(orig_h, y + h)
            right = min(orig_w, x + w)

            # 3. Align face using landmarks on original-resolution frame
            aligned = _align_face(rgb_frame, landmarks_orig)
            if aligned is None:
                continue

            # 4. Encode face
            face_encoding = self._encode_face(aligned)

            # 5. Match against known faces
            name = "Unknown"
            distance = 1.0  # Default high distance (cosine)

            if len(self.known_face_encodings) > 0:
                # Cosine distance = 1 - cosine_similarity
                # Since embeddings are L2-normalized, dot product = cosine similarity
                similarities = np.dot(self.known_face_encodings, face_encoding)
                best_match_index = np.argmax(similarities)
                best_similarity = similarities[best_match_index]
                distance = round(float(1.0 - best_similarity), 2)

                if distance < self.tolerance:
                    name = self.known_face_names[best_match_index]

            score = round(confidence, 2)
            results.append((top, right, bottom, left, name, face_encoding, distance, score))

        return results
