import face_recognition
import numpy as np
import database
import logging
import dlib

logging.basicConfig(level=logging.INFO)

class FacialRecognitionSystem:
    def __init__(self, tolerance: float = 0.6, min_score: float = 0.0):
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance
        self.min_score = min_score # dlib adjust_threshold
        self.detector = dlib.get_frontal_face_detector()
        self.reload_known_faces()

    def reload_known_faces(self):
        """Loads known faces from the database."""
        faces = database.get_known_faces()
        self.known_face_encodings = [np.array(face['encoding']) for face in faces]
        self.known_face_names = [face['name'] for face in faces]
        logging.info(f"Loaded {len(self.known_face_names)} known faces. Tolerance: {self.tolerance}")


            
    def process_frame(self, rgb_frame):
        """
        Detects faces in the frame and identifies them.
        Returns a list of (top, right, bottom, left, name, encoding, distance, score) tuples.
        """
        # 1. Detect Faces using dlib directly to access scores
        try:
            # run(image, upsample_num_times, adjust_threshold)
            # score limit is handled by adjust_threshold (higher = stricter)
            # OPTIMIZATION: upsample=0 drastically reduces CPU usage. 
            dets, scores, idxs = self.detector.run(rgb_frame, 0, self.min_score)
        except Exception as e:
            logging.error(f"Dlib detection failed: {e}")
            return []
        
        face_locations = []
        face_scores = []
        
        for i, det in enumerate(dets):
            # Convert dlib rect to (top, right, bottom, left)
            top = det.top()
            right = det.right()
            bottom = det.bottom()
            left = det.left()
            
            # Clamp to frame
            height, width = rgb_frame.shape[:2]
            top = max(0, top)
            left = max(0, left)
            bottom = min(height, bottom)
            right = min(width, right)
            
            face_locations.append((top, right, bottom, left))
            face_scores.append(scores[i])

        # 2. Encode Faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []

        for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
            name = "Unknown"
            distance = 1.0 # Default High Distance

            # If a match was found, we still want the distance for logging
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                distance = face_distances[best_match_index]
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            # Format distance for easier reading (2 decimal places)
            distance = round(float(distance), 2)
            
            # Get detection score
            score = round(float(face_scores[i]), 2)
            
            results.append((top, right, bottom, left, name, face_encoding, distance, score))

        return results
