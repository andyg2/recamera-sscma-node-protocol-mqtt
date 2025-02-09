import paho.mqtt.client as mqtt
import time
import json
import uuid
import numpy as np
from collections import defaultdict
from collections import deque
import cv2
import base64
message_times = deque(maxlen=100)  # Store last 100 timestamps

class KalmanFilter:
    def __init__(self, initial_state, dt=0.1, process_noise=1.0, measurement_noise=1.0):
        self.state = np.array(initial_state, dtype=np.float32)
        self.dt = dt
        
        # Increase process noise for velocity components
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State transition matrix with more emphasis on velocity
        self.F = np.array([
            [1, 0, dt, 0],    # x = x + dx*dt
            [0, 1, 0, dt],    # y = y + dy*dt
            [0, 0, 1, 0],     # dx = dx
            [0, 0, 0, 1]      # dy = dy
        ], dtype=np.float32)
        
        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance - increased for velocity components
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initial covariance matrix
        self.P = np.array([
            [10, 0, 0, 0],    # High uncertainty in position
            [0, 10, 0, 0],
            [0, 0, 1000, 0],  # Very high uncertainty in velocity
            [0, 0, 0, 1000]
        ], dtype=np.float32)
        
        # Track time since last update
        self.last_update_time = time.time()
    
    def predict(self):
        # Update dt based on actual time elapsed
        current_time = time.time()
        self.dt = current_time - self.last_update_time
        
        # Update state transition matrix with new dt
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        
        # Update process noise covariance with new dt
        self.Q = np.array([
            [self.dt**4/4, 0, self.dt**3/2, 0],
            [0, self.dt**4/4, 0, self.dt**3/2],
            [self.dt**3/2, 0, self.dt**2, 0],
            [0, self.dt**3/2, 0, self.dt**2]
        ], dtype=np.float32) * self.process_noise
        
        # Predict next state
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement):
        # Innovation (measurement residual)
        y = measurement - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        
        # Update timestamp
        self.last_update_time = time.time()
    
    def get_state(self):
        """Return full state including velocity"""
        return self.state
    
    def get_velocity(self):
        """Return current velocity estimate"""
        return self.state[2:]
    
    def get_position_uncertainty(self):
        """Return uncertainty in position estimate"""
        return np.sqrt(np.diag(self.P)[:2])
      

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

def generate_oval_mask(width, height):
    """Creates an oval-shaped quadratic dropoff mask with exact ROI size."""
    Y, X = np.ogrid[:height, :width]

    # Compute normalized distances (ellipse formula)
    normalized_x = ((X - width / 2) / (width / 2)) ** 2
    normalized_y = ((Y - height / 2) / (height / 2)) ** 2

    # Quadratic dropoff
    normalized_distance = np.sqrt(normalized_x + normalized_y)
    mask = np.clip(1 - normalized_distance ** 2, 0, 1)

    # Convert to 8-bit grayscale (0-255) and ensure single-channel
    mask = (mask * 255).astype(np.uint8)

    return mask  # Shape: (height, width), dtype: uint8

def predict_trajectory(kf, velocity, time_delta):
    state = kf.get_state()
    return state[:2] + velocity * time_delta


def handle_occlusion(kalman_filters, now, max_age=2.0):
    """Handle temporary occlusions by keeping trackers alive longer"""
    for obj_id, (kf, hist, last_seen, size, velocity) in list(kalman_filters.items()):
        time_unseen = now - last_seen
        if time_unseen > max_age:
            del kalman_filters[obj_id]
        elif time_unseen > 0.5:  # Partially occluded
            # Predict new position but don't update
            kf.predict()
def update_appearance_model(old_hist, new_hist, learning_rate=0.3):
    """Gradually update appearance model"""
    return old_hist * (1 - learning_rate) + new_hist * learning_rate

def model_message_handler(payload):
    global next_id, kalman_filters, message_times
    
    now = time.time()
    message_times.append(now)
    
    if "data" in payload and isinstance(payload["data"], dict) and "boxes" in payload["data"]:
        image = payload["data"]["image"]
        image = base64.b64decode(image)
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        boxes = payload["data"]["boxes"]
        resolution = payload["data"]["resolution"]
        
        # Track matches to prevent duplicate assignments
        used_trackers = set()
        
        for box in boxes:
            x, y, w, h, confidence, class_id = box
            roi = image[y:y + h, x:x + w]
            
            # Add size check tolerance
            area = w * h
            
            # Compute histogram with mask
            mask = generate_oval_mask(w, h)
            if roi.shape[:2] != mask.shape:
                mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
            histogram = cv2.calcHist([roi], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            histogram = cv2.normalize(histogram, histogram).flatten()
            
            center_x = x + (w / 2)
            center_y = y + (h / 2)
            measurement = np.array([center_x, center_y])
            
            matched_id = None
            min_score = float('inf')
            
            # Scoring system for matching
            for obj_id, (kf, obj_hist, last_seen, obj_size, velocity) in kalman_filters.items():
                if obj_id in used_trackers:
                    continue
                    
                predicted_position = kf.predict()
                
                # Calculate various similarity metrics
                distance = np.linalg.norm(measurement - predicted_position)
                hist_score = compare_histograms(histogram, obj_hist)
                size_ratio = abs(area - obj_size) / max(area, obj_size)
                time_since_last_seen = now - last_seen
                
                # Predicted position based on velocity
                expected_position = predicted_position + velocity * time_since_last_seen
                velocity_score = np.linalg.norm(measurement - expected_position)
                
                # Combined score with weights
                total_score = (
                    distance * 0.4 +  # Position weight
                    hist_score * 0.3 +  # Appearance weight
                    size_ratio * 0.2 +  # Size consistency weight
                    velocity_score * 0.1  # Motion prediction weight
                )
                
                # Thresholds for matching
                if (distance < w/4 and  # Stricter distance threshold
                    hist_score < 0.75 and 
                    size_ratio < 0.3 and  # Allow 30% size difference
                    time_since_last_seen < 1.0):  # Recent enough
                    
                    if total_score < min_score:
                        min_score = total_score
                        matched_id = obj_id
            
            if matched_id is not None:
                # Update tracker
                kf = kalman_filters[matched_id][0]
                current_velocity = measurement - kf.get_state()[:2]
                
                # Update Kalman filter
                kf.update(measurement)
                
                # Exponential moving average for velocity
                alpha = 0.7  # Smoothing factor
                new_velocity = alpha * current_velocity + (1 - alpha) * kalman_filters[matched_id][4]
                
                # Update tracker info
                kalman_filters[matched_id] = (
                    kf, 
                    histogram,  # Update histogram
                    now,  # Update timestamp
                    area,  # Update size
                    new_velocity  # Update velocity
                )
                used_trackers.add(matched_id)
            else:
                # Create new tracker
                matched_id = next_id
                next_id += 1
                kalman_filters[matched_id] = (
                    KalmanFilter(np.array([center_x, center_y, 0, 0])),
                    histogram,
                    now,
                    area,
                    np.array([0, 0])  # Initial velocity
                )
            
            print(f"Entry {payload['data']['count']}: ID {matched_id}, Score {min_score if matched_id in used_trackers else 'new'}")
            
            
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
