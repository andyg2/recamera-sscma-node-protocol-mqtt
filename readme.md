#### A simple Python script to simulate controlling the SSCMA node with Node-RED.

Based on the kind work of [LynnL4](https://github.com/Seeed-Studio/sscma-example-sg200x/commits?author=LynnL4)

[https://github.com/Seeed-Studio/sscma-example-sg200x/blob/main/test/test_node.py]()

This code has not been fully tested and is just a basic example program - as such, it may require further adjustments to work as expected.

### Furthermore this is just an experimental repository with frequent breaking changes. Use at your own risk.

Experimental additions:

- Track each entity individually: KalmanFilter and Quadratic Masked Histogram
- Timers/Warnings to ensure we keep up with the MQTT messaging

To get it running, you'll need to do the following:

```text
Disable the recamera Node-RED service:
sudo mv /etc/init.d/S03node-red /etc
```

Allow local network access to the recamera device's MQTT broker:

`sudo vi /etc/mosquitto/mosquitto.conf`

Add Lines:

```text
listener 1883
allow_anonymous true
```

Install Python and the `phao-mqtt ` `numpy` and `cv2` package in your environment:

`pip install phao-mqtt numpy cv2`

Modify the IP address of the camera at the end of recamera-mqtt.py

```text
client.start("192.168.88.232", 1883)
```

Run

`python recamera-mqtt.py`
