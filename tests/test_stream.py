import unittest
import queue
import json
import time
import threading
import sys
import os

sys.path.append(os.getcwd())
from stream import StreamManager

class TestStreamManager(unittest.TestCase):
    def setUp(self):
        self.stream_manager = StreamManager()

    def test_singleton(self):
        # StreamManager itself isn't a singleton pattern implementation, 
        # but the app usage enforces it. Here we test the class logic.
        sm1 = StreamManager()
        sm2 = StreamManager()
        self.assertNotEqual(sm1, sm2)

    def test_subscribe(self):
        """Test client subscription."""
        gen = self.stream_manager.subscribe()
        
        # subscribe() blocks at q.get(), so we need to run next(gen) in a separate thread
        # or verify clients count logic differently.
        # But clients are added *before* loop. But next() executes until yield.
        # To get to yield, we need q.get() to return.
        
        # Workaround: Put something in queue? We can't access queue yet.
        # So we use a thread.
        def start_gen():
            try:
                next(gen)
            except StopIteration:
                pass

        t = threading.Thread(target=start_gen)
        t.start()
        
        # Allow thread to reach q.get()
        time.sleep(0.1)
        
        self.assertEqual(len(self.stream_manager.clients), 1)
        
        # Unblock thread
        self.stream_manager.publish_event({'foo': 'bar'})
        t.join(timeout=1.0)
        
        # Explicit cleanup (generator closing) happens when we stop iterating or GC.
        # Here we just verify it was added and we unblocked it.
        # To simulate disconnect, we can close generator
        gen.close()
        # Closing generator raises GeneratorExit in thread? 
        # If thread is blocked in q.get(), close() might not work until it yields.
        # But we made it yield once with publish_event.
        # If we call next(gen) again, it blocks.
        
        # Verify removal
        # clients.remove(q) is in finally/except block.
        # gen.close() triggers GeneratorExit at yield point.
        # The generator is currently yielded (after next() returned).
        # So close() should work.
        gen.close()
        self.assertEqual(len(self.stream_manager.clients), 0)

    def test_publish_event_format(self):
        """Test that published events are formatted correctly as SSE data."""
        # Create a mock client queue
        client_q = queue.Queue()
        self.stream_manager.clients.append(client_q)
        
        test_event = {"type": "test_event", "data": "foo"}
        self.stream_manager.publish_event(test_event)
        
        msg = client_q.get(timeout=1)
        # format: {"type": "event", "data": {event_dict}}
        parsed = json.loads(msg)
        self.assertEqual(parsed['type'], 'event')
        self.assertEqual(parsed['data'], test_event)

    def test_publish_stats_format(self):
        """Test that published stats are formatted correctly."""
        client_q = queue.Queue()
        self.stream_manager.clients.append(client_q)
        
        test_stats = {"cam1": {"fps": 30}}
        self.stream_manager.publish_stats(test_stats)
        
        msg = client_q.get(timeout=1)
        parsed = json.loads(msg)
        self.assertEqual(parsed['type'], 'stats')
        self.assertEqual(parsed['data'], test_stats)

if __name__ == '__main__':
    unittest.main()
