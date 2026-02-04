
import sys
import unittest
import os
from unittest.mock import MagicMock

# Global mocks needed for imports to work at all
sys.modules['face_recognition'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['dlib'] = MagicMock()

# Now runs tests
if __name__ == '__main__':
    # Add CWD to path
    sys.path.append(os.getcwd())
    
    # We must be careful about shared mocks.
    # test_camera_logic patches stream, which is fine.
    
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    
    sys.exit(not result.wasSuccessful())
