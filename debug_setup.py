import sys
import os

def check_and_fix():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check ONNX models
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    models = {
        'face_detection_yunet_2023mar.onnx': 'YuNet face detection',
        'w600k_r50.onnx': 'ArcFace face recognition'
    }
    
    all_ok = True
    for filename, description in models.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"SUCCESS: {description} model found ({size_mb:.1f}MB)")
        else:
            print(f"ERROR: {description} model NOT found at {path}")
            all_ok = False
    
    # Check onnxruntime
    try:
        import onnxruntime as ort
        print(f"SUCCESS: onnxruntime {ort.__version__} installed")
    except ImportError:
        print("ERROR: onnxruntime NOT installed. Run: pip install onnxruntime")
        all_ok = False
    
    # Check OpenCV (for YuNet FaceDetectorYN)
    try:
        import cv2
        print(f"SUCCESS: OpenCV {cv2.__version__} installed")
        if hasattr(cv2, 'FaceDetectorYN'):
            print("SUCCESS: cv2.FaceDetectorYN available")
        else:
            print("WARNING: cv2.FaceDetectorYN not available (need OpenCV >= 4.5.4)")
            all_ok = False
    except ImportError:
        print("ERROR: OpenCV NOT installed")
        all_ok = False
    
    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks failed. See errors above.")

if __name__ == "__main__":
    check_and_fix()
