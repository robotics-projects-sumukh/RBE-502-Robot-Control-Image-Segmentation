import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DepthImageProcessor(Node):
    def __init__(self):
        super().__init__('depth_image_processor')
        
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/simple_drone/front/depth/image_raw',
            self.listener_callback,
            10
        )
        
    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Example pixel coordinates
        x_p, y_p = 320, 250
        
        # Get depth value for the pixel
        depth_value = depth_image[y_p, x_p]
        
        self.get_logger().info(f'Distance at pixel ({x_p}, {y_p}) is {depth_value} meters')

def main(args=None):
    rclpy.init(args=args)
    node = DepthImageProcessor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
