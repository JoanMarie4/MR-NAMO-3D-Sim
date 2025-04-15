import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import numpy as np
import sys
sys.path.append(r"C:\IsaacLab\IsaacLab_git\apps\mr_namo\namosim")
from namosim.mapgen.mapgen import MapGen


def GRID_TO_WORLD(x, y, cell_size):
    return x * cell_size, y * cell_size


def create_mr_namo_map(map_grid, cell_size=0.5):
    for r in range(map_grid.shape[0]):
        for c in range(map_grid.shape[1]):
            if map_grid[r, c] in (1, 2):
                world_x, world_y = GRID_TO_WORLD(c, r, cell_size)
                prim_path = f"/World/Objects/Wall_{r}_{c}"

                prim_utils.create_prim(
                    prim_path,
                    "Cube",
                    scale=(cell_size, cell_size, cell_size),
                    translation=(world_x, world_y, cell_size / 2)
                )


def design_scene():
    """Designs the scene by spawning ground plane, light, objects, and MR-NAMO map."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # Create a new xform prim for all objects
    prim_utils.create_prim("/World/Objects", "Xform")

    mapgen = MapGen(40, 40, init_open=0.42)
    map_grid = mapgen.gen_map()
    create_mr_namo_map(map_grid)

    # Spawn a usd file of a chair into the scene
    cfg_chair = sim_utils.UsdFileCfg(usd_path=f"C:/IsaacLab/Assets/ArchVis/Commercial/Seating/Newman.usd", 
                                     scale=(0.01, 0.01, 0.01))
    cfg_chair.func("/World/Objects/Chair", cfg_chair, translation=(0.15, 2.5, -2.5))

