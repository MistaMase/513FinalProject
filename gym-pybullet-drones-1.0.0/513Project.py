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
import quaternion

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from CPSControllerDynamics import CPSControllerDynamics
from CPSTrajectoryGeneration import CPSTrajectory
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
    parser.add_argument('--duration_sec',       default=20,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0., 0., 1.]])
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

    # Testing Simply Trajectory
    simple_traj = False
    if simple_traj:
        waypoints = np.array([[0., 0., 2],
                              [-1., 0., 2],
                              [-1., -1., 1.0]])
        waypoints_rotation = np.array([quaternion.from_euler_angles(0, 0, 0),
                                       quaternion.from_euler_angles(0, 0, 0),
                                       quaternion.from_euler_angles(0, 0, 0)])
        time_per_waypoint = np.array([2., 2., 2.])
        traj_gen = CPSTrajectory(INIT_XYZS[0], waypoints, ARGS.control_freq_hz, time_per_waypoint, waypoints_rotation)
        traj_pos, traj_rot = traj_gen.linear_interpolation_quaternion()
        # Change back to Euler Angles only for the existing controller
        traj_rot = quaternion.as_euler_angles(traj_rot)

    else:
        waypoints = np.array([[0., 0., 1.5, 0., 0., 0.],
                              [1., 0., 1.5, 0., 0, 0.],
                              [1., 0., 1.75, np.pi/2., 0., 0.],
                              [1., 0., 2.0, np.pi, 0., 0.],
                              [1., 0., 1.75, np.pi/2., 0., 0.],
                              [1., 0., 1.5, 0., 0, 0.]])
        time_per_waypoint = np.array([2., 2., 0.5, 0.5, 0.5, 0.5])
        traj_gen = CPSTrajectory(INIT_XYZS[0], waypoints, ARGS.control_freq_hz, time_per_waypoint, is_quaternion=False)
        traj_pos, traj_rot = traj_gen.linear_interpolation()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=1,
                    duration_sec=ARGS.duration_sec
                    )

    #### Initialize the controllers ############################
    #ctrl_main = CPSControllerDynamics(drone_model=ARGS.drone)
    ctrl_main = DSLPIDControl(drone_model=ARGS.drone)

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str('0'): np.array([0, 0, 0, 0])}
    START = time.time()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        drone_state = obs[str(0)]["state"]
        # Observation Vector: X, Y, Z, Q1, Q2, Q3, Q4, Roll, Pitch, Yaw, Velocity X, Velocity Y, Velocity Z,
        # Angular Velocity X, Angular Velocity Y, Angular Velocity Z, Power 0, Power 1, Power 2, Power 3

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            # Follows our trajectory while there's data
            try:
                rpms, pos_error, yaw_error = ctrl_main.computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                    state=drone_state,
                                                                    target_pos=traj_pos[int(i/CTRL_EVERY_N_STEPS)],
                                                                    target_rpy=traj_rot[int(i/CTRL_EVERY_N_STEPS)])

            # Hovers when we've finished the trajectory
            except IndexError:
                if not simple_traj:
                    rpms, pos_error, yaw_error = ctrl_main.computeControlFromState(
                        control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                        state=drone_state,
                        target_pos=waypoints[-1, 0:3],
                        target_rpy=waypoints[-1, 3:6])
                else:
                    rpms, pos_error, yaw_error = ctrl_main.computeControlFromState(
                        control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                        state=drone_state,
                        target_pos=waypoints[-1],
                        target_rpy=quaternion.as_euler_angles(waypoints_rotation[-1]))

            # Update the action values
            action['0'] = rpms

        #### Log the simulation ####################################
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=drone_state,
                   control=np.zeros(12) # TODO Not Sure What Data Should Go In Here
                   #control=np.append(traj_pos, traj_rot, INIT_XYZS[0], np.zeros(9))
                   #control=np.hstack([TARGET_POS[wp_counters[0], :], INIT_XYZS[0 ,2], np.zeros(9)])
                   )

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
    logger.save_as_csv("513FP") # Optional CSV save

    #### Plot the simulation results ###########################
    logger.plot()
