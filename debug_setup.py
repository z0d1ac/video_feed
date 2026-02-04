import sys
import subprocess
import os

def check_and_fix():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    try:
        import face_recognition_models
        print("SUCCESS: face_recognition_models is installed and importable.")
        print(f"Location: {os.path.dirname(face_recognition_models.__file__)}")
    except ImportError:
        print("ERROR: face_recognition_models NOT found.")
        print("Attempting to install it now using the current python environment...")
        
        try:
            # Install directly using the current python executable
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/ageitgey/face_recognition_models"
            ])
            print("\nInstallation attempted. Trying import again...")
            
            import face_recognition_models
            print("SUCCESS: face_recognition_models is now installed!")
        except subprocess.CalledProcessError as e:
            print(f"\nInstallation failed with error code: {e.returncode}")
        except ImportError:
            print("\nFAILURE: Installed successfully but still cannot import. This is weird.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    check_and_fix()
