import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IsaacLab")))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Test G1 Demo in rough terrain environment."
)
from namosim.simulator import Simulator, create_sim_from_file
# append RSL-RL cli argumentsks
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner
import namosim.navigation.action_result as ar
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
scenarios_folder = os.path.join(os.path.dirname(__file__), "../scenarios")
from omni.physx.scripts import utils as physx_utils
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
from flat_env_namo_cfg import G1FlatMultiRobotEnvCfg


TASK = "Isaac-Velocity-Flat-G1-v0"
RL_LIBRARY = "rsl_rl"

class G1RoughDemo:
        
    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        # load the trained jit policy
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
        # create envionrment
        simulation_file_path=os.path.join(
            scenarios_folder, "evasion.svg"
        )
        self.sim = create_sim_from_file(simulation_file_path)
        self.world = self.sim.ref_world
        self.agents = self.world.agents
        self.agent_to_env = {name: idx for idx, name in enumerate(self.agents)}
        self.binary_grid = self.world.map
        self.cell_size = self.binary_grid.cell_size
        self.spawn_pos = self.compute_spawn_poses(self.agents, self.binary_grid)
        env_cfg = G1FlatMultiRobotEnvCfg()
        env_cfg.scene.num_envs = len(self.agents)
        env_cfg.curriculum = None 
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)

        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        # load prev trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self.generate_3d_map_from_occupancy_grid(self.binary_grid)
        self.commands = torch.zeros(env_cfg.scene.num_envs, 4, device=self.device)
        #self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

    def generate_3d_map_from_occupancy_grid(self, binary_grid):
        """Creates 3D obstacles from the binary occupancy grid."""
        stage = omni.usd.get_context().get_stage()
        walls_path = "/World/Walls"
        obstacle_group = UsdGeom.Xform.Define(stage, walls_path)
        for x in range(binary_grid.d_width):
            for y in range(binary_grid.d_height):
                if binary_grid.grid[x, y] > 0:  # Obstacle detected in this cell
                    self.create_box_at_cell(stage, obstacle_group.GetPath().AppendChild(f"Cube_{x}_{y}"), x, y, binary_grid)

        self.create_boundary_walls(stage, binary_grid)

        #mergnig cubes into one mesh\
        omni.kit.commands.execute(
            "MergeMeshPrims",
            sourcePrimPaths=["/World/Walls"],
            destPrimPath="/World/MergedWalls",
            createNewLayer=False
        )

    def create_boundary_walls(self, stage, binary_grid):
        """Creates 4 surrounding boundary walls around the map."""
        cell_size = binary_grid.cell_size
        scale = 1.5
        width = binary_grid.d_width * cell_size * scale
        height = binary_grid.d_height * cell_size * scale
        wall_thickness = cell_size * scale
        wall_height = cell_size * 20

        offset_x = binary_grid.grid_pose[0] * scale
        offset_y = binary_grid.grid_pose[1] * scale

        positions = [
            # Bottom wall
            ("Boundary_Bottom", (offset_x + width / 2, offset_y - wall_thickness / 2, wall_height / 2), (width, wall_thickness, wall_height)),
            # Top wall
            ("Boundary_Top", (offset_x + width / 2, offset_y + height + wall_thickness / 2, wall_height / 2), (width, wall_thickness, wall_height)),
            # Left wall
            ("Boundary_Left", (offset_x - wall_thickness / 2, offset_y + height / 2, wall_height / 2), (wall_thickness, height, wall_height)),
            # Right wall
            ("Boundary_Right", (offset_x + width + wall_thickness / 2, offset_y + height / 2, wall_height / 2), (wall_thickness, height, wall_height)),
        ]

        for name, pos, size in positions:
            path = f"/World/Walls/{name}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.GetSizeAttr().Set(1.0)
            xform = UsdGeom.XformCommonAPI(cube.GetPrim())
            xform.SetTranslate(Gf.Vec3d(*pos))
            xform.SetScale(Gf.Vec3f(size[0], size[1], size[2]))


    def create_box_at_cell(self, stage, prim_path, x, y, binary_grid):
        """Creates a scaled cube obstacle in the world at the given grid cell."""
        scale = 1.5
        world_x = (x * binary_grid.cell_size + binary_grid.cell_size / 2 + binary_grid.grid_pose[0]) * scale
        world_y = (y * binary_grid.cell_size + binary_grid.cell_size / 2 + binary_grid.grid_pose[1]) * scale
        wall_height = binary_grid.cell_size * 20  # Taller than 1 cell

        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.GetSizeAttr().Set(binary_grid.cell_size)

        xform = UsdGeom.XformCommonAPI(cube.GetPrim())
        xform.SetTranslate(Gf.Vec3d(world_x, world_y, wall_height / 2))
        xform.SetScale(Gf.Vec3f(scale, scale, wall_height / binary_grid.cell_size))



    def compute_spawn_poses(self, agents, binary_grid):
        spawn_poses = []
        scale = 1.5
        for agent in agents.values():
            # Convert grid cell indices to meters
            print(f"Raw pose: {agent.pose}")
            print(f"Grid pose origin: {binary_grid.grid_pose}")
            print(f"Cell size: {binary_grid.cell_size}")
            print(f"Agent Cell size: {agent.cell_size}")
            x = agent.pose[0] / binary_grid.cell_size
            y = agent.pose[1] / binary_grid.cell_size
            world_x = (x * binary_grid.cell_size + binary_grid.cell_size / 2 + binary_grid.grid_pose[0]) * scale
            world_y = (y * binary_grid.cell_size + binary_grid.cell_size / 2 + binary_grid.grid_pose[1]) * scale
            yaw_rad = agent.pose[2]
            spawn_poses.append((world_x, world_y, yaw_rad))
        return spawn_poses

    '''
    def set_agent_spawns(self, spawn_poses):
        """Sets the spawn poses of all robots using the world pose."""
        for env_index, (x, y, yaw) in enumerate(spawn_poses):
            quat = euler_angles_to_quat([0.0, 0.0, yaw])

            # Get the robot in this env
            robot = self.env.unwrapped.scene[f"robot_{env_index}"]
            # Set the world pose of this robot
            robot.set_world_pose(
                position=[x, y, 0.0],  # Assuming robot starts on flat ground
                orientation=quat,
            )
    '''

    def set_agent_spawns(self, spawn_positions):

        robot = self.env.unwrapped.scene["robot"]
        device = robot.data.root_pos_w.device

        if len(spawn_positions) != self.env.num_envs:
            raise ValueError(
                f"Number of spawn positions ({len(spawn_positions)}) does not match number of environments ({self.env.num_envs})."
            )

        #update the root state position for each robot in each environment
        for env_index, pos in enumerate(spawn_positions):
            # Keep orientation unchanged, just update position
            pos_tensor = torch.tensor(pos, dtype=torch.float, device=device)
            robot.data.root_pos_w[env_index] = pos_tensor

        #apply the updated states to simulation
        robot.write_root_state_to_sim(robot.data.root_state_w)
        
    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = omni.usd.get_context().get_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 1
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if event.input.name in self._key_to_control:
                if self._selected_id:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
            # Escape key exits out of the current selected robot view
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        """Determines which robot is currently selected and whether it is a valid G1 robot.
        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None 
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a G1 robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")


    
    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

def main():
    """Main function."""
    demo_g1 = G1RoughDemo()
    obs, _ = demo_g1.env.reset()
    # Delay the manual position update by one frame to allow Isaac Sim to stabilize
    demo_g1.set_agent_spawns(demo_g1.spawn_pos)
    history = demo_g1.sim.run()
    while simulation_app.is_running():
        for step in history:
            action_results = step.action_results
            for agent_id, action_result in action_results.items():
                env_index = demo_g1.agent_to_env.get(agent_id)

                if env_index is None:
                    raise KeyError(f"Unknown agent_id '{agent_id}' in results")
                
                if isinstance(action_result, ar.ActionSuccess):
                    robot_pose = action_result.robot_pose
                    # now you can use robot_pose + env_index however you like
                    print(f"[env {env_index}] {agent_id} pose = {robot_pose}")
                else:
                    # it's some other ActionResult (e.g. a failure or no-op), skip or handle differently
                    print(f"[env {env_index}] {agent_id} did not succeed, skipping pose extraction")
        # check for selected robots
        demo_g1.update_selected_object()
        with torch.inference_mode():
            action = demo_g1.policy(obs)
            obs, _, _, _ = demo_g1.env.step(action)
            # overwrite command based on keyboard input
            #obs[:, 9:13] = demo_g1.commands
            if demo_g1._selected_id is not None:
                obs[demo_g1._selected_id, 9:13] = demo_g1.commands[demo_g1._selected_id]
                obs[torch.arange(obs.shape[0]) != demo_g1._selected_id, 9:13] = 0.0
            else:
                obs[:, 9:13] = 0.0



if __name__ == "__main__":
    main()
    simulation_app.close()

