import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import cv2
import torch
import numpy as np
from yolact import Yolact  # Import YOLACT model
from torchvision.transforms import functional as F
from cv_bridge import CvBridge

class YOLACTDetector(Node):
    def __init__(self):
        super().__init__('yolact_detector')
        
        # Initialize YOLACT model
        self.model = Yolact()
        self.model.load_weights('/home/sumukh/RBE502_Robot_Control/drone_ws/src/yolact/weights/yolact_base_54_800000.pth')  # Path to the YOLACT model weights
        self.model.eval()
        
        # ROS2 subscribers and publishers
        self.image_subscriber = self.create_subscription(
            Image, '~/front/image_raw', self.image_callback, 10)
        self.target_publisher = self.create_publisher(
            Float64MultiArray, 'target', 10)
        
        self.cv_bridge = CvBridge()  # To convert between ROS Image and OpenCV Image
        
        # Class names for YOLACT (replace with the actual YOLACT class labels)
        self.class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle']  # Fill in the rest of the classes

    def image_callback(self, msg):
        # Convert ROS2 Image message to OpenCV image
        frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Preprocess the image for YOLACT
        frame_tensor = F.to_tensor(frame).unsqueeze(0)
        with torch.no_grad():
            preds = self.model(frame_tensor)  # Run YOLACT
        
        # Extract bounding boxes, class predictions, and scores
        classes, scores, boxes, masks = preds

        # Look for 'person' class (assuming class ID 1 corresponds to 'person')
        person_class_id = self.class_names.index('person')
        
        person_found = False
        for i in range(len(classes)):
            if classes[i] == person_class_id:
                # Person detected, get the bounding box
                x1, y1, x2, y2 = boxes[i]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Create and publish Float64MultiArray for target position
                target_msg = Float64MultiArray()
                target_msg.data = [center_x, center_y]
                self.target_publisher.publish(target_msg)
                self.get_logger().info(f'Published target coordinates: ({center_x}, {center_y})')
                
                person_found = True
                break
        
        if not person_found:
            self.get_logger().info('No person detected.')

def main(args=None):
    rclpy.init(args=args)
    yolact_detector = YOLACTDetector()
    
    try:
        rclpy.spin(yolact_detector)
    except KeyboardInterrupt:
        pass
    
    yolact_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
