import numpy as np
import quaternion

class CPSTrajectory:
    # Start Location:   [X, Y, Z, Q0, Q1, Q2, Q3]
    # Waypoints:        [X, Y, Z, Q0, Q1, Q2, Q3]
    def __init__(self, start_location, waypoints, update_freq, time_per_waypoint, waypoints_quaternion=None, is_quaternion=True):
        self.is_quaternion = is_quaternion
        self.update_freq = update_freq
        self.time_per_waypoint = time_per_waypoint
        self.waypoints = waypoints
        self.waypoints_quaternion = waypoints_quaternion

        if is_quaternion:
            self.start_location = start_location
            self.start_rotation = np.quaternion(1, 0, 0, 0)
            self._trajectory_position = np.zeros((int(np.sum(self.time_per_waypoint) * self.update_freq), 3))
            self._trajectory_rotation = np.zeros(int(np.sum(self.time_per_waypoint) * self.update_freq), dtype=np.quaternion)
        else:
            self.start_location = np.append(start_location, [0., 0., 0.])
            self._trajectory = np.zeros((int(np.sum(self.time_per_waypoint) * self.update_freq), 6))

    def linear_interpolation(self):
        if self.is_quaternion:
            print(f'This function should not be used with quaternions')

        # Connect Start to First Waypoint
        start_num = 0
        num_pts = int(self.update_freq*self.time_per_waypoint[0])
        distance = self.waypoints[0] - self.start_location
        step = np.linspace(0., 1., num=num_pts, endpoint=False)
        for i in range(0, num_pts):
            self._trajectory[start_num+i] = self.start_location + step[i]*distance
        start_num += num_pts

        # Connect each waypoint to one another
        for i in range(0, len(self.waypoints)-1):
            num_pts = int(self.update_freq*self.time_per_waypoint[i+1])
            distance = self.waypoints[i+1] - self.waypoints[i]
            if i == len(self.waypoints)-2:
                step = np.linspace(0., 1., num=num_pts, endpoint=True)
            else:
                step = np.linspace(0., 1., num_pts, endpoint=False)
            for j in range(0, num_pts):
                self._trajectory[start_num+j] = self.waypoints[i] + step[j]*distance
            start_num += num_pts

        print(f'Trajectory: {self._trajectory}')
        return self._trajectory[:, 0:3], self._trajectory[:, 3:6]

    def linear_interpolation_quaternion(self):
        # Connect Start to First Waypoint
        start_num = 0
        num_pts = int(self.update_freq * self.time_per_waypoint[0])

        # Position
        distance = self.waypoints[0] - self.start_location
        step = np.linspace(0., 1., num=num_pts, endpoint=False)

        for i in range(0, num_pts):
            self._trajectory_position[start_num + i] = self.start_location + step[i] * distance
        start_num += num_pts

        # Rotation
        self._trajectory_rotation[start_num:start_num+num_pts] = quaternion.squad(
            np.array([self.start_rotation, self.waypoints_quaternion[0]]),
            np.array([0, num_pts]),
            np.arange(num_pts))

        # Connect each waypoint to one another
        for i in range(0, len(self.waypoints) - 1):
            num_pts = int(self.update_freq * self.time_per_waypoint[i + 1])

            # Position
            distance = self.waypoints[i + 1] - self.waypoints[i]
            if i == len(self.waypoints) - 2:
                step = np.linspace(0., 1., num=num_pts, endpoint=True)
            else:
                step = np.linspace(0., 1., num_pts, endpoint=False)
            for j in range(0, num_pts):
                self._trajectory_position[start_num + j] = self.waypoints[i] + step[j] * distance

            # Rotation
            if i == len(self.waypoints) - 2:
                self._trajectory_rotation[start_num:start_num + num_pts] = quaternion.squad(
                    np.array([self.waypoints_quaternion[i + 1], self.waypoints_quaternion[i]]),
                    np.array([0, num_pts-1]),
                    np.arange(num_pts))
            else:
                self._trajectory_rotation[start_num:start_num + num_pts] = quaternion.squad(
                    np.array([self.waypoints_quaternion[i + 1], self.waypoints_quaternion[i]]),
                    np.array([0, num_pts]),
                    np.arange(num_pts))
            start_num += num_pts

        print(f'Trajectory')
        for pos, rot in zip(self._trajectory_position, self._trajectory_rotation):
            print(f'Position: {pos}\tRotation: {rot}')
        return self._trajectory_position, self._trajectory_rotation

    def cost_optimation(self):
        pass


if __name__ == '__main__':
    start_location = np.array([0., 0., 1.])
    waypoints = np.array([[0., 0., 2.],
                          [1., 0., 2.],
                          [1., 1., 2.]
                         ])
    update_freq = 5.
    time_per_waypoint = np.array([2., 2., 2.])
    traj = CPSTrajectory(start_location, waypoints, update_freq, time_per_waypoint)
    traj.linear_interpolation()
