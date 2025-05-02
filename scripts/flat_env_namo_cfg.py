
import os
import sys
import omni
import omni.isaac.kit
import omni.usd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IsaacLab")))
from source.isaaclab_tasks.isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from source.isaaclab_tasks.isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards, G1RoughEnvCfg

from isaaclab.utils import configclass

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import source.isaaclab_tasks.isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from namosim.simulator import Simulator, create_sim_from_file
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from namosim.simulator import Simulator, create_sim_from_file
from namosim.mapgen.mapgen import MapGen
import namosim.agents as agts
from namosim import agents
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import UsdGeom, Gf
scenarios_folder = os.path.join(os.path.dirname(__file__), "../scenarios")

@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class G1FlatMultiRobotEnvCfg(G1FlatEnvCfg):
    scenario_file: str = os.path.join(
            scenarios_folder, "social_dr_success_d.svg"
        )

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 2.5  # Space between each robot/environment
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.enabled = False
        
    def generate_3d_map_from_occupancy_grid(self, binary_grid):
        """Creates 3D obstacles from the binary occupancy grid."""
        stage = omni.usd.get_context().get_stage()
        root_path = "/World/Map"
        for x in range(binary_grid.d_width):
            for y in range(binary_grid.d_height):
                if binary_grid.grid[x, y] > 0:  # Obstacle detected in this cell
                    self.create_box_at_cell(stage, root_path, x, y, binary_grid)

    def create_box_at_cell(self, stage, root_path, x, y, binary_grid):
        """Creates a scaled cube obstacle in the world at the given grid cell."""
        world_x = x * binary_grid.cell_size + binary_grid.cell_size / 2 + binary_grid.grid_pose[0]
        world_y = y * binary_grid.cell_size + binary_grid.cell_size / 2 + binary_grid.grid_pose[1]
        wall_height = binary_grid.cell_size * 10  # Taller than 1 cell

        cube_path = f"{root_path}/Obstacle_{x}_{y}"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(binary_grid.cell_size)

        xform = UsdGeom.XformCommonAPI(cube.GetPrim())
        xform.SetTranslate(Gf.Vec3d(world_x, world_y, wall_height / 2))
        xform.SetScale(Gf.Vec3f(1.0, 1.0, wall_height / binary_grid.cell_size))

    def compute_spawn_poses(self, agents):
        spawn_poses = []
        for agent in agents.values():
            x_meters = agent.pose[0] * agent.cell_size
            y_meters = agent.pose[1] * agent.cell_size
            yaw_rad = agent.pose[2]
            spawn_poses.append((x_meters, y_meters, yaw_rad))
        return spawn_poses

    def reset_robots_to_agents(self, scene, spawn_poses):
        """Reset robots to their respective spawn positions."""
        for env_index, (x, y, yaw) in enumerate(spawn_poses):
            quat = euler_angles_to_quat([0.0, 0.0, yaw])
            
            # Access the robot associated with this env_index
            robot = scene["robot"].get(env_index)
            
            # Set world pose of this robot instance
            robot.set_world_pose(
                position=[x, y, 0.0],
                orientation=quat,
            )
