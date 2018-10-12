import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Trajectory:

    def __init__(self, speed, init_pos, orientation, 
            final_pos, final_orientation):

        self.__i = 0
        self.__init_orientation = math.pi * orientation / 180.0
        self.__targ_orientation = math.pi * final_orientation / 180.0
        self.__init_pos = np.array([init_pos]).T
        self.__targ_pos = np.array([final_pos]).T
        self.__waypoints = []

        self.__speed = speed 

        self.add_waypoint(init_pos[0], init_pos[1])
        self.add_waypoint(final_pos[0], final_pos[1])

        self.__init_u = np.array([
            [1 * math.cos(math.pi * orientation / 180.0)],
            [1 * math.sin(math.pi * orientation / 180.0)]
        ])

        self.__init_v = np.array([
            [speed * math.cos(math.pi * orientation / 180.0)],
            [speed * math.sin(math.pi * orientation / 180.0)]
        ])

    def next(self):
        self.__i = self.__i + 1

    def add_waypoint(self, x, y):
        self.__waypoints.append(np.array([[x], [y]]))

    def param_a(self):

        curr_v = self.unit_velocity_at_point(self.__i)
        next_v = self.unit_velocity_at_point(self.__i + 1)

        time_estimate = self.estimate_time_between_points(self.__speed, self.__i)

        a = (next_v + curr_v) * time_estimate
        a = a - 2 * (self.__waypoints[self.__i + 1] -\
                     self.__waypoints[self.__i])
        a = a * 6
        a = a / (time_estimate ** 3)

        return a

    def param_b(self):

        curr_v = self.unit_velocity_at_point(self.__i)
        next_v = self.unit_velocity_at_point(self.__i + 1)

        time_estimate = self.estimate_time_between_points(self.__speed, self.__i)

        b = (next_v + 2 * curr_v) * time_estimate
        b = b - 3 * (self.__waypoints[self.__i + 1] -\
                     self.__waypoints[self.__i])
        b = -2 * b
        b = b / (time_estimate ** 2)

        return b

    def motion(self, t):

        a = self.param_a()
        b = self.param_b()

        v0 = self.unit_velocity_at_point(self.__i)

        v = 0.5 * a * (t ** 2) + b * (t) + v0

        w = (a[1] * t + b[1]) * v[0] - (a[0] * t + b[0]) * v[1]
        w = w / ((v[0] ** 2) + (v[1] ** 2))

        forward_speed = np.linalg.norm(v)
        angular_speed = float(w)

        return forward_speed, angular_speed

    def frange(x, y, step):
        while x < y:
            yield x
            x += step

    def plot_trajectory(self, delta_time):

        temp = self.__i
        
        straight_line_x = []
        straight_line_y = []

        vector_at_point_x = []
        vector_at_point_y = []

        # Draw the unit velocity at each waypoint.
        for i in range(len(self.__waypoints)):
            straight_line_x.append(self.__waypoints[i][0])
            straight_line_y.append(self.__waypoints[i][1])

            velocity_at_point = self.unit_velocity_at_point(i)

            wp = self.__waypoints[i]

            p0 = wp - velocity_at_point
            p1 = wp + velocity_at_point

            vector_at_point_x.append([p0[0], p1[0]])
            vector_at_point_y.append([p0[1], p1[1]])

        segment_line_x = []
        segment_line_y = []

        t0 = 0
        time_estimate = self.estimate_time_between_points(
            self.__speed, self.__i)
        end_time = time_estimate

        dis_x = self.__waypoints[0][0]
        dis_y = self.__waypoints[0][1]
        curve_line_x = [dis_x]
        curve_line_y = [dis_y]
        segment_line_x.append(curve_line_x[-1])

        theta = self.__init_orientation
        timestep = 1
        while True:
            if timestep >= end_time:
                segment_line_x.append(curve_line_x[-1])
                self.next()

                if self.__i == len(self.__waypoints) - 1:
                    break

                time_estimate = self.estimate_time_between_points(
                    self.__speed, self.__i)
                end_time = end_time + time_estimate
                t0 = timestep

            v, w = self.motion(timestep - t0)
            theta = (theta + w)
            dis_x = dis_x + v * delta_time * math.cos(theta)
            dis_y = dis_y + v * delta_time * math.sin(theta)
            curve_line_x.append(dis_x)
            curve_line_y.append(dis_y)

            timestep += delta_time

        # Plotting.

        plt.plot(straight_line_x, straight_line_y, 'bo-')
        plt.plot(curve_line_x, curve_line_y, 'kx--')

        for i in range(len(vector_at_point_x)):
            plt.plot(vector_at_point_x[i], vector_at_point_y[i], 'c--')

        for x_coords in segment_line_x:
            plt.axvline(x=x_coords, linestyle='--', color='gray')

        init_pos = (float(self.__init_pos[0]), float(self.__init_pos[1]))
        init_orientation = 180.0 * self.__init_orientation / math.pi
        targ_pos = (float(self.__targ_pos[0]), float(self.__targ_pos[1]))
        targ_orientation = 180.0 * self.__targ_orientation / math.pi
        plt.title('Calculated trajectory with initial position %s and '\
                'orientation %s degree, final position %s and orientation %s'\
                ' degree.'
                % (init_pos, init_orientation, targ_pos,\
                    targ_orientation))
        plt.show()

        self.__i = temp

    def estimate_time_between_points(self, speed, i):

        next_line = self.__waypoints[i + 1] - self.__waypoints[i]
        next_line_norm = np.linalg.norm(next_line)

        curr_v = self.unit_velocity_at_point(i)
        curr_v_norm = np.linalg.norm(curr_v)

        next_v = self.unit_velocity_at_point(i + 1)
        next_v_norm = np.linalg.norm(next_v)

        theta1 = np.dot(next_line.flatten(), curr_v.flatten()) /\
                (next_line_norm * curr_v_norm)
        theta1 = math.acos(theta1)

        if theta1 == 0:
            theta1 = 0.00001

        len1 = abs((abs(next_line_norm) * theta1) / math.sin(theta1)) 

        theta2 = np.dot(next_line.flatten(), next_v.flatten()) /\
                (next_line_norm * next_v_norm)
        theta2 = math.acos(theta2)

        if theta2 == 0:
            theta2 = 0.00001

        len2 = abs((abs(next_line_norm) * theta2) / math.sin(theta2))

        speed1 = np.linalg.norm(speed * curr_v)
        speed2 = np.linalg.norm(speed * next_v)

        estimate_time = (len1 + len2) / (speed1 + speed2)

        return estimate_time
        
    def unit_velocity_at_point(self, i):

        if i == 0:
            return self.__init_u

        if i == len(self.__waypoints) - 1:
            return np.array([
                [math.cos(self.__targ_orientation)],
                [math.sin(self.__targ_orientation)]
            ])

        p_curr = self.__waypoints[i]
        p_prev = self.__waypoints[i - 1]
        p_next = self.__waypoints[i + 1]

        # The direction vector pointing in the direction from p_prev to p_curr
        # (a), and from p_curr to p_next (b).
        a = p_curr - p_prev
        b = p_next - p_curr

        # Get the unit vector at point p_curr.
        m = (a * np.linalg.norm(b) + b * np.linalg.norm(a))
        u = m / np.linalg.norm(m)

        return u

    def waypoint_str(self):
    
        s = ''
        for wp in self.__waypoints:
            x = float(wp[0])
            y = float(wp[1])

            s += str((x, y)) + ', '
        return s
