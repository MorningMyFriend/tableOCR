import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
textdetection_hori_path = osp.join(this_dir, 'text-detection-ctpn')
add_path(textdetection_hori_path)

# Add caffe to PYTHONPATH
textdetection_angle_path = osp.join(this_dir, 'textdetection_angle')
add_path(textdetection_angle_path)
