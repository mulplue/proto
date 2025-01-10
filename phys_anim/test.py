# from phys_anim.envs.location.isaacgym import LocationHumanoid
import sys
# sys.path.append("/home/chenjiahe/3_hsi/proto")

# sys.path.pop(0)
# sys.path.pop(3)
# sys.path.pop(-2)
# sys.path.pop(-1)
# sys.path = []
# sys.path = sys.path[-1]
print(sys.path)

# import numpy

# import phys_anim.scripts
# print(phys_anim.scripts)
# # from phys_anim.scripts import play_motion
# import phys_anim.scripts.aa
# import phys_anim.scripts.bb
# # from phys_anim.scripts import play_motion
# # from phys_anim.envs import steering_1
# # from phys_anim.envs import base_interface

from isaac_utils.isaac_utils import rotations
from isaac_utils.isaac_utils.rotations import quat_rotate, quat_from_angle_axis, normalize_angle
print(rotations)
# from isaac_utils.rotations import quat_rotate, quat_from_angle_axis, normalize_angle