import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Path to your rosbag
cam_num = 2  # 2 is left, 3 is right, 1 is front
bag_num = 55
# bag_path = f"data/makalii_point/cam{cam_num}/bag_camera_{cam_num}_2025_08_13-01_35_58_{bag_num}/bag_camera_{cam_num}_2025_08_13-01_35_58_{bag_num}.db3"
bag_path = "/home/kalliyanlay/Documents/BYU/research/camera_lidar_calibration/data/airbnb_outside/bag_camera_2_2025_08_08-20_59_48_0.db3"
# Topics
depth_topic = f"/oak_d_lr_{cam_num}/stereo/image_raw"
rgb_topic = f"/oak_d_lr_{cam_num}/rgb/image_raw"
print("Playing topics:", rgb_topic, "and", depth_topic)

# Setup bag reader
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)
reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

bridge = CvBridge()

# Keep track of last RGB frame
last_rgb_frame = None

while reader.has_next():
    topic, data, t = reader.read_next()

    if topic == rgb_topic:
        msg = deserialize_message(data, Image)
        last_rgb_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    elif topic == depth_topic:
        msg = deserialize_message(data, Image)
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # Normalize depth for visualization
        if depth_image.dtype != np.uint8:
            depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)
        else:
            depth_display = depth_image

        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # Combine RGB and depth side by side if RGB is available
        if last_rgb_frame is not None:
            # Resize depth to match RGB height if needed
            if depth_colormap.shape[:2] != last_rgb_frame.shape[:2]:
                depth_colormap = cv2.resize(depth_colormap, (last_rgb_frame.shape[1], last_rgb_frame.shape[0]))
            combined = cv2.hconcat([last_rgb_frame, depth_colormap])
        else:
            combined = depth_colormap

        cv2.imshow("RGB + Depth", combined)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
