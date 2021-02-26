import os
import sys


def clean_ros_opencv():
    CHECK_ROS_OPENCV2 = "/opt/ros/kinetic/lib/python2.7/dist-packages"
    if CHECK_ROS_OPENCV2 in sys.path:
        sys.path.remove(CHECK_ROS_OPENCV2)    
    return


def disable_tensorflow_logging():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return


def clean():
    clean_ros_opencv()
    disable_tensorflow_logging()
    return
