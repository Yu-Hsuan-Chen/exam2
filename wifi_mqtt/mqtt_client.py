import paho.mqtt.client as paho
import serial
import time
import re

serdev = '/dev/ttyACM0'
s = serial.Serial(serdev, 9600)


# MQTT broker hosted on local machine
mqttc = paho.Client()

# Settings for connection
# TODO: revise host to your IP
host = "192.168.160.34"
topic = "Mbed"
# Callbacks
def on_connect(self, mosq, obj, rc):
    print("Connected rc: " + str(rc))

def on_message(mosq, obj, msg):
    if(re.search('The', str(msg.payload))):
        print("[Received] Topic : " + msg.topic + " , " + msg.payload.decode('utf-8') + "\n")
    elif(re.search('\d+', str(msg.payload))):
        print("[Received] Topic : " + msg.topic  + "  " + ", Threshold Angle : " + msg.payload.decode('utf-8') + "\n")
        time.sleep(2)
        angle = msg.payload.decode('utf-8')
        s.write(bytes(f"/detectionMode/run {angle}\r", 'UTF-8'))
        time.sleep(1)
    else:
        print("[Received] Topic : " + msg.topic + ", Message: " + msg.payload.decode('utf-8') + "\n")

def on_subscribe(mosq, obj, mid, granted_qos):
    print("Subscribed OK")

def on_unsubscribe(mosq, obj, mid, granted_qos):
    print("Unsubscribed OK")

# Set callbacks
mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.on_subscribe = on_subscribe
mqttc.on_unsubscribe = on_unsubscribe

# Connect and subscribe
print("Connecting to " + host + "/" + topic)
mqttc.connect(host, port=1883, keepalive=60)
mqttc.subscribe(topic, 0)

# Publish messages from Python
num = 0
while num != 3:
    ret = mqttc.publish(topic, "Message from Python!\n", qos=0)
    if (ret[0] != 0):
            print("Publish failed")
    mqttc.loop()
    time.sleep(1.5)
    num += 1

time.sleep(2)
s.write(bytes("/gestureMode/run\r", 'UTF-8'))
time.sleep(1)

# Loop forever, receiving messages
mqttc.loop_forever()
