"""Script demonstrating the implementation of the downwash effect model.

Example
-------
In a terminal, run as:

    $ python downwash.py

Notes
-----
The drones move along 2D trajectories in the X-Z plane, between x == +.5 and -.5.

"""
import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Downwash example script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--duration_sec',       default=12,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0., 0., 0.]])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1
    env = CtrlAviary(drone_model=ARGS.drone,
                     num_drones=1,
                     initial_xyzs=INIT_XYZS,
                     physics=Physics.PYB_DW,
                     neighbourhood_radius=10,
                     freq=ARGS.simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=ARGS.gui,
                     record=ARGS.record_video,
                     obstacles=True
                     )

    #### Initialize the trajectories ###########################
    PERIOD = 5
    LIFTOFF_HEIGHT = 1 # m
    LIFTOFF_TIME = 2 # s
    NUM_WP = ARGS.control_freq_hz*LIFTOFF_TIME
    TARGET_POS = np.zeros((NUM_WP, 3))
    TARGET_POS[:, 2] = np.arange(NUM_WP) * (LIFTOFF_HEIGHT/NUM_WP)
    # for i in range(NUM_WP):
    #     TARGET_POS[i, :] = [0.5*np.cos(2*np.pi*(i/NUM_WP)), 0, 0]
    print(f'Target Pos: {TARGET_POS}')

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=1,
                    duration_sec=ARGS.duration_sec
                    )

    #### Initialize the controllers ############################
    #ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(1)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(1)}
    START = time.time()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        # if i%CTRL_EVERY_N_STEPS == 0:

            # #### Compute control for the current way point #############
            # #for j in range(2):
            # action[str(0)], _, _ = ctrl[0].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
            #                                                            state=obs[str(0)]["state"],
            #                                                            target_pos=np.hstack([INIT_XYZS[0, 2], TARGET_POS[wp_counters[0], :]]),
            #                                                            )
            # print(f'HStack: {np.hstack([TARGET_POS[wp_counters[0], :], INIT_XYZS[0, 2]])}')
            #
            # #### Go to the next way point and loop #####################
            # #for j in range(2):

        #### Log the simulation ####################################
        #for j in range(2):
        # logger.log(drone=0,
        #            timestamp=i/env.SIM_FREQ,
        #            state=obs[str(0)]["state"],
        #            control=np.hstack([TARGET_POS[wp_counters[0], :], INIT_XYZS[0 ,2], np.zeros(9)])
        #            )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("dw") # Optional CSV save

    #### Plot the simulation results ###########################
    logger.plot()
