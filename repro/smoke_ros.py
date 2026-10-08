#!/usr/bin/env python2
"""Isolated ROS image -> original YOLO node -> detection/image smoke test.

Source the rebuilt catkin workspace, then run with /usr/bin/python2.
Starts its own ROS master on port 11321; does not control robot hardware.
"""
from __future__ import print_function
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--classes', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    probe = socket.socket()
    # Ignore TIME_WAIT from a previous test, while still rejecting a live listener.
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(('127.0.0.1', 11321))
    finally:
        probe.close()
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:11321'
    os.environ['ROS_HOSTNAME'] = '127.0.0.1'
    os.environ.pop('ROS_IP', None)
    os.environ['ROS_LOG_DIR'] = os.path.join(args.output, 'roslogs')
    processes, logs = [], []
    try:
        master_log = open(os.path.join(args.output, 'master.log'), 'w')
        logs.append(master_log)
        processes.append(subprocess.Popen(['roscore', '-p', '11321'], stdout=master_log, stderr=subprocess.STDOUT, preexec_fn=os.setsid))
        import xmlrpclib
        master = xmlrpclib.ServerProxy(os.environ['ROS_MASTER_URI'])
        deadline = time.time() + 30
        while True:
            try:
                if master.getPid('/backup_smoke')[0] == 1:
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise RuntimeError('Isolated ROS master did not start')
            time.sleep(0.2)
        import rospy
        import cv2
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        from bot_vision.msg import DetectionArray
        rospy.init_node('backup_smoke', anonymous=True)
        received = {'detections': None, 'image': None}
        def detections(msg):
            received['detections'] = msg
        def image(msg):
            received['image'] = msg
        sub_det = rospy.Subscriber('/detections', DetectionArray, detections)
        sub_img = rospy.Subscriber('/detection_result', Image, image)
        pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
        node_log = open(os.path.join(args.output, 'node.log'), 'w')
        logs.append(node_log)
        node = subprocess.Popen(['stdbuf', '-oL', '-eL', args.node, '__name:=yolo_trt_node', '_engine_path:='+args.engine, '_class_names_str:='+args.classes, '_conf_thresh:=0.3'], stdout=node_log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        processes.append(node)
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError('Cannot read sample image')
        frame = cv2.resize(frame, (640, 480))
        bridge = CvBridge()
        deadline = time.time() + 60
        while time.time() < deadline:
            if node.poll() is not None:
                raise RuntimeError('YOLO node exited; see node.log')
            msg = bridge.cv2_to_imgmsg(frame, 'bgr8')
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)
            if received['detections'] is not None and received['image'] is not None:
                break
            time.sleep(0.5)
        if received['detections'] is None or received['image'] is None:
            raise RuntimeError('No detection/image response within 60 seconds')
        labels = args.classes.split(',')
        rows = []
        for d in received['detections'].detections:
            values = [d.prob, d.x, d.y, d.w, d.h]
            if d.label not in labels or any(math.isnan(v) or math.isinf(v) for v in values):
                raise RuntimeError('Invalid detection')
            if not (0 <= d.prob <= 1 and d.w >= 0 and d.h >= 0):
                raise RuntimeError('Invalid probability or box')
            rows.append(dict(label=d.label, prob=d.prob, x=d.x, y=d.y, w=d.w, h=d.h))
        result = received['image']
        if (result.width, result.height, result.encoding) != (640, 480, 'bgr8'):
            raise RuntimeError('Unexpected visualization shape/encoding')
        # Empty detections are valid, but model-load success must be independently visible.
        node_log.flush()
        log_text = open(os.path.join(args.output, 'node.log')).read()
        if 'Model loaded successfully' not in log_text or 'MODEL NOT LOADED' in log_text:
            raise RuntimeError('Model-load confirmation absent from node log')
        cv2.imwrite(os.path.join(args.output, 'result.jpg'), bridge.imgmsg_to_cv2(result, 'bgr8'))
        report = dict(engine=args.engine, classes=labels, input_image=args.image, image_size=[640, 480], detections=rows, passed=True, scope='single-image pipeline smoke; not an accuracy benchmark')
        with open(os.path.join(args.output, 'result.json'), 'w') as out:
            json.dump(report, out, indent=2)
        print(json.dumps(report))
        rospy.signal_shutdown('smoke complete')
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                deadline = time.time() + 5
                while process.poll() is None and time.time() < deadline:
                    time.sleep(0.1)
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
