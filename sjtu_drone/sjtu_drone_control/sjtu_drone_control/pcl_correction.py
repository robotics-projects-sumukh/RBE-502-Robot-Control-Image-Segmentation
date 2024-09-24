import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudTransformer(Node):
    def __init__(self):
        super().__init__('point_cloud_transformer')
        
        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            '/simple_drone/front/points',
            self.listener_callback,
            10
        )
        
        # Publisher
        self.publisher = self.create_publisher(PointCloud2, '/simple_drone/front/corrected_points', 10)

        # Transformation angles (in degrees)
        self.angle_x = 0  # X-axis rotation angle
        self.angle_y = 90 # Y-axis rotation angle
        self.angle_z = -90 # Z-axis rotation angle

    def listener_callback(self, msg):
        # Convert PointCloud2 message to numpy array
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Convert to a list of points and then to a NumPy array
        points = np.array(list(pc_data))
        
        if points.size == 0:
            self.get_logger().warn('Received empty point cloud data')
            return

        # Check if points is 1-dimensional and reshape if necessary
        if points.ndim == 1:
            points = np.vstack((points['x'], points['y'], points['z'])).T
        else:
            # Convert from structured array to a plain 2D array
            points = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float64)
        
        # Apply transformations
        transformed_points = self.apply_transformations(points)
        
        # Create new PointCloud2 message
        header = msg.header
        new_msg = pc2.create_cloud_xyz32(header, transformed_points)
        
        # Publish the transformed points
        self.publisher.publish(new_msg)

    def apply_transformations(self, points):
        # Convert angles from degrees to radians
        rad_x = np.deg2rad(self.angle_x)
        rad_y = np.deg2rad(self.angle_y)
        rad_z = np.deg2rad(self.angle_z)
        
        # Rotation matrices
        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x), np.cos(rad_x)]
        ])
        
        rotation_y = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ])
        
        rotation_z = np.array([
            [np.cos(rad_z), -np.sin(rad_z), 0],
            [np.sin(rad_z), np.cos(rad_z), 0],
            [0, 0, 1]
        ])
        
        # Apply rotations
        points_transformed = points @ rotation_x.T @ rotation_z.T @ rotation_y.T
        
        return points_transformed

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudTransformer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
