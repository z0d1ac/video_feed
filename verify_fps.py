import time
import json
import sys
from stream import RTSPStream

def test_fps(target_fps):
    print(f"\n------------------------------------------------")
    print(f"Testing Stream with Target FPS: {target_fps}")
    print(f"------------------------------------------------")
    
    # Load config for source URL
    try:
        with open('config.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found.")
        return

    source = config.get('video_source')
    if not source:
        print("Error: video_source not found in config.")
        return

    # Initialize Stream with target FPS
    stream = RTSPStream(source, max_fps=target_fps)
    stream.start()
    
    print("Waiting for stream to stabilize (up to 15s)...")
    for _ in range(150):
        if stream.grabbed:
            break
        time.sleep(0.1)

    if not stream.grabbed:
        print("Error: Stream failed to start or grab frames.")
        stream.stop()
        return

    print("Counting frames...")
    count = 0
    start_time = time.time()
    last_frame_id = -1
    
    # Measure for 10 seconds for accuracy
    duration = 10.0
    end_time = start_time + duration
    
    while time.time() < end_time:
        grabbed, frame, frame_id = stream.read()
        
        if grabbed and frame_id != last_frame_id:
            count += 1
            last_frame_id = frame_id
            # Print a dot every frame to show activity
            sys.stdout.write('.')
            sys.stdout.flush()
            
        time.sleep(0.005) # Poll frequently
        
    print("\n")
    stream.stop()
    
    actual_duration = time.time() - start_time
    actual_fps = count / actual_duration
    
    print(f"Results:")
    print(f"  Target FPS: {target_fps}")
    print(f"  Frames Captured: {count}")
    print(f"  Duration: {actual_duration:.2f}s")
    print(f"  Actual FPS: {actual_fps:.2f}")
    
    error_margin = abs(target_fps - actual_fps)
    if error_margin < 2.0:
        print("  ✅ SUCCESS: FPS matches target within margin.")
    else:
        print("  ⚠️ VARIANCE: Actual FPS differs significantly from target.")

if __name__ == "__main__":
    print("Starting FPS Verification Test...")
    # Test a low FPS to verify limiting works
    test_fps(5)
    # Test the user's current setting (usually 10)
    test_fps(10)
