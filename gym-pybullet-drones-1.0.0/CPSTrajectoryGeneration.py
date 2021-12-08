import numpy as np


class CPSTrajectory:
    # Start Location:   [X, Y, Z, Q0, Q1, Q2, Q3]
    # Waypoints:        [X, Y, Z, Q0, Q1, Q2, Q3]
    def __init__(self, start_location, waypoints, update_freq, time_per_waypoint):
        self.start_location = start_location
        self.waypoints = waypoints
        self.update_freq = update_freq
        self.time_per_waypoint = time_per_waypoint
        self._trajectory = np.zeros((int(np.sum(time_per_waypoint)*update_freq), 3))

    def linear_interpolation(self):
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

        # Add the last point

        print(f'Trajectory: {self._trajectory}')
        return self._trajectory

    def linear_interpolation_smoothing(self):
        pass

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
