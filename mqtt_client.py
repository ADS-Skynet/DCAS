"""
mqtt_client.py
Simple MQTT publish wrapper for the DCAS driver monitoring system.

Broker runs on the Jetson (vehicle side). PC connects as client.
Topics:
    dcas/driver/alert  — immediate alert on distraction trigger
    dcas/driver/state  — VLM classification result
"""

import paho.mqtt.client as mqtt

TOPIC_ALERT = "dcas/driver/alert"
TOPIC_STATE = "dcas/driver/state"


class MQTTClient:
    def __init__(self, broker: str, port: int = 1883):
        self._broker = broker
        self._port = port
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._connected = False

    def connect(self):
        try:
            self._client.connect(self._broker, self._port, keepalive=60)
            self._client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection failed ({self._broker}:{self._port}): {e}")

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    def publish_alert(self, message: str):
        """Publish to dcas/driver/alert (immediate distraction trigger)."""
        self._publish(TOPIC_ALERT, message)

    def publish_state(self, message: str):
        """Publish to dcas/driver/state (VLM classification result)."""
        self._publish(TOPIC_STATE, message)

    def _publish(self, topic: str, message: str):
        if not self._connected:
            print(f"[MQTT] Not connected — skipped: {topic} = {message!r}")
            return
        result = self._client.publish(topic, message, qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] Publish failed: rc={result.rc}, topic={topic}")
        else:
            print(f"[MQTT] Published: {topic} = {message!r}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[MQTT] Connected to {self._broker}:{self._port}")
        else:
            print(f"[MQTT] Connect refused: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect: rc={rc}")
