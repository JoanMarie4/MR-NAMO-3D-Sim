
import argparse
from isaaclab.app import AppLauncher
import os

# Set environment variables to disable GUI and RViz
os.environ["NAMO_NO_DISPLAY_WINDOW"] = "TRUE"
os.environ["NAMO_DEACTIVATE_RVIZ"] = "TRUE"
#from isaaclab.robots.unitree import G1_CFG  # Import G1 config

# create argparser
parser = argparse.ArgumentParser(description="MR-NAMO Scene 1 (IN PROGRESS)")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from environment import setup_world, create_mr_namo_environment

import isaaclab.sim as sim_utils

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by adding assets to it
    setup_world()
    create_mr_namo_environment()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()