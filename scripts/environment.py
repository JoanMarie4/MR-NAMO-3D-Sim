import os
import sys
import omni
import omni.isaac.kit
import omni.usd
from pxr import UsdGeom, Gf
import numpy as np

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import your custom modules
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from namosim.simulator import Simulator, create_sim_from_file
from namosim.mapgen.mapgen import MapGen

from isaacsim.core.utils.rotations import euler_angles_to_quat

# Where your scenarios/maps are located
scenarios_folder = os.path.join(os.path.dirname(__file__), "../scenarios")

def setup_world():
    """Create ground, lights, and prepare scene."""
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    cfg_light = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))

    prim_utils.create_prim("/World/Objects", "Xform")

def create_mr_namo_environment():
    """Create map and spawn robots based on scenario."""
    sim = create_sim_from_file(
        simulation_file_path=os.path.join(
            scenarios_folder, "social_dr_success_d.svg"
        )
    )
    world = sim.ref_world
    bin_occ_grid = world.map
    agents = world.agents

    generate_3d_map_from_occupancy_grid(bin_occ_grid)

    #spawn_poses = compute_spawn_poses(agents)

    # Reset robots
    #reset_robots_to_agents(world.scene, spawn_poses)

def generate_3d_map_from_occupancy_grid(binary_grid: BinaryOccupancyGrid):
    """Creates 3D obstacles from the binary occupancy grid."""
    stage = omni.usd.get_context().get_stage()
    root_path = "/World/Map"

    for x in range(binary_grid.d_width):
        for y in range(binary_grid.d_height):
            if binary_grid.grid[x, y] > 0:
                create_box_at_cell(stage, root_path, x, y, binary_grid)

def create_box_at_cell(stage, root_path, x, y, binary_grid):
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

'''
def compute_spawn_poses(agents):
    spawn_poses = []
    for agent in agents:
        x_meters = agent.pose[0] * agent.cell_size
        y_meters = agent.pose[1] * agent.cell_size
        yaw_rad = agent.pose[2]
        spawn_poses.append((x_meters, y_meters, yaw_rad))
    return spawn_poses

def reset_robots_to_agents(scene, spawn_poses):
    for env_index, (x, y, yaw) in enumerate(spawn_poses):
        quat = euler_angles_to_quat([0.0, 0.0, yaw])
        scene.envs[env_index].robot.set_world_pose(
            position=[x, y, 0.0],
            orientation=quat,
        )
'''
