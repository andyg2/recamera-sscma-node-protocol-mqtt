import paho.mqtt.client as mqtt
import time
import json
import uuid
import numpy as np
from collections import defaultdict


import numpy as np
import cv2
import base64

import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, dt=1.0, process_noise=1.0, measurement_noise=1.0):
        self.state = np.array(initial_state, dtype=np.float32)  # Ensure it's always 4D
        self.dt = dt  # Time step
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State transition matrix
        self.F = np.array([[1, 0, dt, 0], 
                           [0, 1, 0, dt], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]], dtype=np.float32)

        # Measurement matrix (position only)
        self.H = np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0.01, 0],
                           [0, 0, 0, 0.01]], dtype=np.float32)

        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        # Covariance matrix (set high for better stability)
        self.P = np.eye(4, dtype=np.float32) * 500

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position (x, y)

    def update(self, measurement):
        y = measurement - self.H @ self.state  # ✅ Corrected line
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y  # Update full state
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

        
        
# Function to convert bounding box to grid number
def convert_to_grid(box, resolution):
    grid_size = 68
    x, y, w, h, _, _ = box

    # Calculate feet position
    feet_x = x + (w / 2)
    feet_y = y + h  # Feet are at the bottom of the bounding box

    # Normalize to grid coordinates
    grid_x = int((feet_x / resolution[0]) * grid_size)
    grid_y = int((feet_y / resolution[1]) * grid_size)

    # Ensure the coordinates are within the grid bounds
    grid_x = max(0, min(grid_x, grid_size - 1))
    grid_y = max(0, min(grid_y, grid_size - 1))

    # Convert to grid number
    grid_number = grid_y * grid_size + grid_x + 1  # +1 to match your numbering
    return grid_number


# Dictionary to store Kalman filters and unique IDs for each tracked object
kalman_filters = {}
next_id = 0  # Counter for assigning unique IDs


class SSCMANodeClient:
    def __init__(self, id="recamera", version="v0"):
        self.id = id
        self.version = version
        self._nodes = []
        self._connected = False
        self._pending_requests = []
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"Connected to MQTT broker! Client ID: {self.id}")
            self.mqtt_client.subscribe(f"sscma/{self.version}/{self.id}/node/out/#")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            id = msg.topic.split("/")[-1]
            for node in self._nodes:
                if node.id == id:
                    node.receive(payload)
                    break
        except json.JSONDecodeError:
            print(
                f"Invalid JSON payload received on topic {msg.topic}: {msg.payload.decode()}"
            )

    def request(self, node, action, data):

        if node != None:
            topic = f"sscma/{self.version}/{self.id}/node/in/{node.id}"
        else:
            topic = f"sscma/{self.version}/{self.id}/node/in/"

        if not self._connected:
            self._pending_requests.append((node, action, data))
            return

        pyload = {"type": 3, "name": action, "data": data}
        print("{}:{}".format(topic, pyload))
        self.mqtt_client.publish(topic, json.dumps(pyload))

    def start(self, broker, port):
        self.mqtt_client.connect(broker, port)
        self.mqtt_client.loop_start()
        while not self._connected:
            pass

        self.request(None, "clear", {})

        for node in self._nodes:
            node.create()

        for request in self._pending_requests:
            self.request(*request)

    def stop(self):
        for node in self._nodes:
            node.destroy()

        self.request(None, "clear", {})

        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        self._connected = False


class Node:
    def __init__(self, client, id=None):
        self.client = client
        self.id = id if id else str(uuid.uuid4())
        self.dependencies = []
        self.dependents = []
        self.onReceive = None
        self.client._nodes.append(self)

    def sink(self, target_node):
        self.dependents.append(target_node)
        target_node.dependencies.append(self)

    def send(self, data):
        for dependent in self.dependents:
            dependent.receive(data)

    def request(self, action, data):
        self.client.request(self, action, data)

    def receive(self, data):
        if self.onReceive:
            self.onReceive(data)

    def destroy(self):
        self.request("destroy", {})

    def enable(self):
        self.request("enabled", True)

    def disable(self):
        self.request("enabled", False)


class CameraNode(Node):
    def __init__(self, client, option="2", audio=True, preview=False):
        super().__init__(client)
        self.option = option
        self.audio = audio
        self.preview = preview

    def create(self):
        data = {
            "type": "camera",
            "config": {
                "option": self.option,
                "audio": self.audio,
                "preview": self.preview,
            },
            "dependencies": [n.id for n in self.dependencies],
            "dependents": [n.id for n in self.dependents],
        }
        self.request("create", data)


class ModelNode(Node):
    def __init__(
        self,
        client,
        uri="",
        tscore=0.45,
        tiou=0.35,
        topk=0,
        labels=None,
        debug=True,
        audio=True,
        trace=False,
        counting=False,
        splitter=None,
    ):
        super().__init__(client)
        self.uri = uri
        self.tscore = tscore
        self.tiou = tiou
        self.topk = topk
        self.labels = labels or []
        self.debug = debug
        self.audio = audio
        self.trace = trace
        self.counting = counting
        self.splitter = splitter or []

    def create(self):
        data = {
            "type": "model",
            "config": {
                "uri": self.uri,
                "tscore": self.tscore,
                "tiou": self.tiou,
                "topk": self.topk,
                "labels": self.labels,
                "debug": self.debug,
                "audio": self.audio,
                "trace": self.trace,
                "counting": self.counting,
                "splitter": self.splitter,
            },
            "dependencies": [n.id for n in self.dependencies],
            "dependents": [n.id for n in self.dependents],
        }
        self.request("create", data)


class StreamingNode(Node):
    def __init__(
        self,
        client,
        protocol="rtsp",
        port=8554,
        session="live",
        user="admin",
        password="admin",
    ):
        super().__init__(client)
        self.protocol = protocol
        self.port = port
        self.session = session
        self.user = user
        self.password = password

    def create(self):
        data = {
            "type": "stream",
            "config": {
                "protocol": self.protocol,
                "port": self.port,
                "session": self.session,
                "user": self.user,
                "password": self.password,
            },
            "dependencies": [n.id for n in self.dependencies],
            "dependents": [n.id for n in self.dependents],
        }
        self.request("create", data)


class SaveNode(Node):
    def __init__(self, client, storage="local", duration=-1, slice_time=300):
        super().__init__(client)
        self.storage = storage
        self.duration = duration
        self.slice_time = slice_time

    def create(self):
        data = {
            "type": "save",
            "config": {
                "storage": self.storage,
                "duration": self.duration,
                "slice": self.slice_time,
                "enabled": True,
            },
            "dependencies": [n.id for n in self.dependencies],
            "dependents": [n.id for n in self.dependents],
        }
        self.request("create", data)

def compare_histograms(hist1, hist2):
    # Use the Bhattacharyya distance to compare histograms
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def cleanup_old_filters(expiry_time=30):
    """Remove Kalman filters that haven't been updated in `expiry_time` seconds."""
    current_time = time.time()
    expired_keys = [obj_id for obj_id, (_, _, last_seen) in kalman_filters.items() if current_time - last_seen > expiry_time]
    
    for key in expired_keys:
        del kalman_filters[key]
        print(f"Removed ID {key} due to inactivity.")

def model_message_handler(payload):
    global next_id, kalman_filters
    hist_score = 1;
    # Remove old filters before processing new data
    cleanup_old_filters(expiry_time=30)

    if "data" in payload and isinstance(payload["data"], dict) and "boxes" in payload["data"]:
        image = payload["data"]["image"]
        
        # Decode base64 image
        image = base64.b64decode(image)
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        boxes = payload["data"]["boxes"]
        resolution = payload["data"]["resolution"]

        for box in boxes:
            x, y, w, h, confidence, class_id = box
            
            # Extract region of interest and compute histogram
            roi = image[y:y + h, x:x + w]
            
            # histogram format: [0-255, 0-255, 0-255]
            histogram = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            histogram = cv2.normalize(histogram, histogram).flatten()
            print(histogram)
            
            # Compute bounding box center
            center_x = x + (w / 2)
            center_y = y + (h / 2)
            measurement = np.array([center_x, center_y])

            # Try to match with an existing object
            matched_id = None
            min_distance = float('inf')
            best_match_score = float('inf')

            for obj_id, (kf, obj_hist, last_seen) in kalman_filters.items():
                predicted_position = kf.predict()
                distance = np.linalg.norm(measurement - predicted_position)

                # Compare histograms
                hist_score = compare_histograms(histogram, obj_hist)

                # If the object is close and histograms are similar, match it
                if distance < (w/5) and hist_score < 0.75:
                    if distance < min_distance and hist_score < best_match_score:
                        min_distance = distance
                        best_match_score = hist_score
                        matched_id = obj_id

            # If no match, create a new tracking entry
            if matched_id is None:
                matched_id = next_id
                next_id += 1
                kalman_filters[matched_id] = (KalmanFilter(np.array([center_x, center_y, 0, 0])), histogram, time.time())

            # Update Kalman filter with the new measurement
            kalman_filters[matched_id][0].update(measurement)

            # Refresh last seen timestamp
            kalman_filters[matched_id] = (kalman_filters[matched_id][0], histogram, time.time())

            # Convert predicted position to grid coordinates
            predicted_position = kalman_filters[matched_id][0].predict()
            grid_number = convert_to_grid(
                [predicted_position[0], predicted_position[1], w, h, confidence, class_id],
                resolution,
            )

            print(f"Entry with count {payload['data']['count']}: ID {matched_id}, Hist {hist_score}")
# Initialize global variables
next_id = 0
kalman_filters = {}  # Dictionary to store Kalman filters and their histograms


def camera_message_handler(data):
    print(f"Camera data: {data}")


def stream_message_handler(data):
    print(f"Stream data: {data}")


def save_message_handler(data):
    print(f"Save data: {data}")


if __name__ == "__main__":
    client = SSCMANodeClient("recamera", "v0")

    camera = CameraNode(client)
    model = ModelNode(client, tiou=0.25, tscore=0.35)
    stream = StreamingNode(
        client,
        protocol="rtsp",
        port=8554,
        session="live",
        user="admin",
        password="admin",
    )
    saving = SaveNode(client, storage="local", slice_time=300)

    camera.sink(model)  # connect camera to model
    camera.sink(stream)  # connect camera to stream
    camera.sink(saving)  # connect camera to saving

    camera.onReceive = camera_message_handler
    model.onReceive = model_message_handler
    stream.onReceive = stream_message_handler
    saving.onReceive = save_message_handler

    client.start("192.168.88.232", 1883)
    try:
        input()
    finally:
        client.stop()
