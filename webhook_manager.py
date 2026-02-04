import requests
import json
import logging
import threading

logger = logging.getLogger('webhooks')
logger.setLevel(logging.INFO)
# Basic console logging/file logging if needed, or rely on app's main logger

def send_webhook(url, payload):
    """
    Sends a webhook POST request in a separate thread to avoid blocking the main app.
    """
    if not url:
        return

    def _send():
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=5)
            if response.status_code >= 400:
                logger.error(f"Webhook failed: {response.status_code} - {response.text}")
            else:
                logger.info(f"Webhook sent to {url}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    # Fire and forget thread
    t = threading.Thread(target=_send)
    t.daemon = True
    t.start()
