import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        
        # Create a ROS2 publisher
        self.publisher_ = self.create_publisher(Image, '/front/image_raw', 10)
        
        # Set up OpenCV video capture (0 for the default camera)
        self.cap = cv2.VideoCapture(0)
        
        # Set the resolution to 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Bridge between ROS Image messages and OpenCV images
        self.bridge = CvBridge()
        
        # Create a timer to publish frames at a fixed rate (30 FPS in this example)
        timer_period = 1/30  # 30 frames per second
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # Read a frame from the camera
        ret, frame = self.cap.read()
        if ret:
            # Convert the OpenCV frame (BGR) to ROS Image message
            ros_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            
            # Publish the image
            self.publisher_.publish(ros_image_msg)
            self.get_logger().info('Publishing video frame')
        else:
            self.get_logger().error('Failed to capture video frame')

    def destroy_node(self):
        # Release the camera when shutting down the node
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()

    try:
        rclpy.spin(video_publisher)
    except KeyboardInterrupt:
        pass

    # Destroy the node and shut down ROS
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
