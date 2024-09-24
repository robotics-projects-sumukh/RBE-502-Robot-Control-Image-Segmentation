import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import torch
import cv2
from cv_bridge import CvBridge
import sys

# Add yolact directory to Python path
yolact_path = '/home/sumukh/RBE502_Robot_Control/drone_ws/src/yolact'  # Update this to your yolact path
sys.path.append(yolact_path)


import numpy as np
# Monkey patch np.int
if not hasattr(np, 'int'):
    np.int = int

# Import YOLACT-specific libraries
from yolact import Yolact
from data import cfg, set_cfg
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess

class YOLACTNode(Node):
    def __init__(self):
        super().__init__('yolact_node')

        # Load YOLACT model
        set_cfg('yolact_base_config')  # Use your model config here
        self.model = Yolact()
        self.model.load_weights('/home/sumukh/RBE502_Robot_Control/drone_ws/src/yolact/weights/yolact_base_54_800000.pth')  # Update with your weights path
        self.model.eval()
        self.model.cuda()  # Move model to GPU

        # ROS2 subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, '/front/image_raw', self.image_callback, 10)
        self.target_pub = self.create_publisher(
            Float64MultiArray, 'target', 10)
        
        # Bridge for converting ROS Image to OpenCV
        self.bridge = CvBridge()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Get the device from the model parameters
        device = next(self.model.parameters()).device
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).to(device)  # Ensure the tensor is on the correct device

        # Run YOLACT detection
        preds = self.model(frame_tensor)

        # Get image width and height
        h, w, _ = frame.shape

        # Before calling postprocess, handle the missing config attribute
        try:
            # Call postprocess normally
            detections = postprocess(preds, w, h, score_threshold=0.3)
        except AttributeError as e:
            # Check if it's due to 'mask_proto_debug' missing
            if 'mask_proto_debug' in str(e):
                # Handle missing 'mask_proto_debug' attribute (ignore it or provide fallback)
                print("'mask_proto_debug' not found. Using default behavior.")
                cfg.mask_proto_debug = False  # Assign default if needed
                detections = postprocess(preds, w, h, score_threshold=0.3)
            else:
                # Raise the error if it's a different issue
                raise e

        # Continue with your bounding box extraction logic for the person class
        person_class = 0  # Assuming class 0 corresponds to 'person'
        for i in range(detections['class_ids'].shape[0]):
            if detections['class_ids'][i] == person_class:
                # Extract bounding box center coordinates
                bbox = detections['boxes'][i]
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2

                # Publish the center coordinates as Float64MultiArray
                target_msg = Float64MultiArray()
                target_msg.data = [x_center.item(), y_center.item()]
                self.target_publisher.publish(target_msg)
                break


def main(args=None):
    rclpy.init(args=args)
    yolact_node = YOLACTNode()

    try:
        rclpy.spin(yolact_node)
    except KeyboardInterrupt:
        pass

    yolact_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
