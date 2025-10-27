import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

# source ~/ros2_ws/install/setup.sh
# View topics: ros2 bag info cammerged_data/bag_cammerged_2025_08_14-00_56_41_10/bag_camera_2_2025_08_14-00_56_40_10.db3
cam_num = 1 # 2 is left, 3 is right, 1 is front
# bag_path = "/home/kalliyanlay/Documents/BYU/research/hawaii_processing/bag_camera_" + str(cam_num) +"_2025_08_14-00_56_40_30.db3"
bag_path = "bag_camera_1_2025_08_11-22_11_16_0.db3"
USE_RECT = False
USE_COMP = True

fps = 30

if USE_RECT:
    topic_to_play = "/oak_d_lr_" + str(cam_num) +"/rgb/image_rect" # or /oak_d_lr_2/rgb/image_raw
    print("topic:", topic_to_play)

    # Setup bag reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                                    output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    bridge = CvBridge()

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == topic_to_play:
            msg = deserialize_message(data, Image)
            frame = bridge.imgmsg_to_cv2(msg)

            cv2.imshow("OAK-D Video", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if USE_COMP:
    topic_to_play = "/oak_d_lr_" + str(cam_num) +"/rgb/image_raw/compressed"

    # Setup bag reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                                    output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    bridge = CvBridge()

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == topic_to_play:
            msg = deserialize_message(data, CompressedImage)
            frame = bridge.compressed_imgmsg_to_cv2(msg)

            cv2.imshow("OAK-D Compressed Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
