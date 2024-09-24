import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64MultiArray
from yolact_ros2_msgs.msg import Detections, Detection

class PersonTargetNode(Node):
    def __init__(self):
        super().__init__('person_target_node')

        # Create a QoS profile with RELIABLE policy and additional settings
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1  # Depth of the queue
        )

        # Create subscriber for Detections with RELIABLE QoS
        self.subscription = self.create_subscription(
            Detections,
            '/yolact_ros2/detections',
            self.detections_callback,
            qos_profile)
        
        # Create publisher for Float64MultiArray (x, y of bounding box center)
        self.publisher_ = self.create_publisher(Float64MultiArray, 'target', 10)
        
    def detections_callback(self, msg):
        # Loop through each detection
        for detection in msg.detections:
            # Check if the detected object is a person
            if detection.class_name == "person":
                # Get bounding box coordinates
                box = detection.box
                x_center = (box.x1 + box.x2) / 2
                y_center = (box.y1 + box.y2) / 2

                # Create a Float64MultiArray to store center coordinates
                center_coordinates = Float64MultiArray()
                center_coordinates.data = [x_center, y_center]

                # Publish the center coordinates
                self.publisher_.publish(center_coordinates)
                self.get_logger().info(f"Person detected, publishing center: [{x_center}, {y_center}]")

def main(args=None):
    rclpy.init(args=args)
    
    # Create and spin the node
    node = PersonTargetNode()
    rclpy.spin(node)

    # Shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
