import torch
import numpy as np
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from isaac_utils import rotations
from phys_anim.envs.location.common import BaseLocation
from phys_anim.envs.base_task.isaaclab import TaskHumanoid


class LocationHumanoid(BaseLocation, TaskHumanoid):
    def __init__(self, config, device: torch.device, simulation_app):
        super().__init__(config=config, device=device, simulation_app=simulation_app)

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def set_up_scene(self) -> None:
        if not self.headless:
            self._load_marker_asset()
        super().set_up_scene()

    def _load_marker_asset(self):
        location_marker_obj_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/LocationMarker",
            markers={
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 1.0), opacity=0.5
                    ),
                ),
            },
        )

        self.location_markers = VisualizationMarkers(location_marker_obj_cfg)

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        marker_root_pos = self.get_humanoid_root_states()[..., 0:3].clone()
        marker_root_pos[..., 0:2] += self._tar_pos
        marker_root_pos[..., 2] = 0.0

        self.location_markers.visualize(
            translations=marker_root_pos,
        )

    def draw_task(self):
        self._update_marker()
        super().draw_task()