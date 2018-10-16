import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

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

        # Initialize the initial unit vector (the unit vector for the initial
        # velocity at initial point).
        self.__init_u = np.array([
            [math.cos(self.__init_orientation)],
            [math.sin(self.__init_orientation)]
        ])

    def next(self):
        self.__i = self.__i + 1

    def next_waypoint(self):
        return self.__waypoints[self.__i + 1]

    def is_final_waypoint(self):
        return self.__i == len(self.__waypoints) - 1

    def add_waypoint(self, x, y):
        self.__waypoints.append(np.array([[x], [y]]))

    def param_a(self):

        curr_v = self.unit_velocity_at_point(self.__i) * self.__speed
        next_v = self.unit_velocity_at_point(self.__i + 1) * self.__speed

        time_estimate = self.estimate_time_between_points(self.__speed, self.__i)

        a = (next_v + curr_v) * time_estimate
        a = a - 2 * (self.__waypoints[self.__i + 1] -\
                     self.__waypoints[self.__i])
        a = a * 6
        a = a / (time_estimate ** 3)

        return a

    def param_b(self):

        curr_v = self.unit_velocity_at_point(self.__i) * self.__speed
        next_v = self.unit_velocity_at_point(self.__i + 1) * self.__speed

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

        v0 = self.unit_velocity_at_point(self.__i) * self.__speed

        v = 0.5 * a * (t ** 2) + b * (t) + v0

        w = (a[1] * t + b[1]) * v[0] - (a[0] * t + b[0]) * v[1]
        w = w / ((v[0] ** 2) + (v[1] ** 2))

        return np.linalg.norm(v), float(w)

    def plot(self, delta_time):

        # Waypoints.
        waypoint_x = [] 
        waypoint_y = []
        for wp in self.__waypoints:
            waypoint_x.append(wp[0])
            waypoint_y.append(wp[1])

        # Velocity at each waypoint.
        vel_at_pt_x = []
        vel_at_pt_y = []
        for i, wp in enumerate(self.__waypoints):

            vel = self.unit_velocity_at_point(i)

            p0 = wp - vel
            p1 = wp + vel

            vel_at_pt_x.append([p0[0], p1[0]]) 
            vel_at_pt_y.append([p0[1], p1[1]]) 

        t0 = 0
        time_estimate = self.estimate_time_between_points(
            self.__speed, self.__i)
        end_time = time_estimate

        pos_x = self.__waypoints[0][0]
        pos_y = self.__waypoints[0][1]
        displacement_x = [pos_x]
        displacement_y = [pos_y]
        theta = self.__init_orientation

        timestep = delta_time

        plt.ion()

        # Init fig.
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=[0, 100], ylim=[0, 100])
        line1, = ax.plot(displacement_x, displacement_y, 'r-')

        while True:

            if timestep >= time_estimate:
                self.next()

                if self.__i == len(self.__waypoints) - 1:
                    break

                time_estimate = self.estimate_time_between_points(
                    self.__speed, self.__i)
                end_time = end_time + time_estimate
                t0 = timestep

            v, w = self.motion(timestep - t0)
            theta = theta + w
            pos_x = pos_x + v * delta_time * math.cos(theta)
            pos_y = pos_y + v * delta_time * math.sin(theta)
            displacement_x.append(pos_x)
            displacement_y.append(pos_y)

            line1.set_xdata(displacement_x)
            line1.set_ydata(displacement_y)
            fig.canvas.draw()
            fig.canvas.flush_events()

            timestep += delta_time
            time.sleep(delta_time)

    def plot_trajectory(self):

        temp = self.__i

        delta_time = 1
        
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

        t0 = 0
        time_estimate = self.estimate_time_between_points(
            self.__speed, self.__i)
        end_time = time_estimate

        dis_x = self.__waypoints[0][0]
        dis_y = self.__waypoints[0][1]
        curve_line_x = [dis_x]
        curve_line_y = [dis_y]

        theta = self.__init_orientation
        timestep = 0
        while True:
            if timestep >= end_time:
                self.next()

                if self.__i == len(self.__waypoints) - 1:
                    break
                
                time_estimate = self.estimate_time_between_points(
                    self.__speed, self.__i)
                end_time = end_time + time_estimate
                t0 = timestep

            v, w = self.motion(timestep - t0)
            d_x = 0
            d_y = 0

            theta = theta + w
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

    def plot_motion(self):

        temp = self.__i

        time_estimate = self.estimate_time_between_points(
            self.__speed, self.__i)
        end_time = time_estimate

        t0 = 0
        timestep = 0
        delta_time = 0.5

        v_plot = []
        w_plot = []
        t_plot = []

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        while True:
            if timestep >= end_time:
                
                self.next()

                if self.__i == len(self.__waypoints) - 1:
                    break

                time_estimate = self.estimate_time_between_points(
                    self.__speed, self.__i)
                end_time = end_time + time_estimate
                t0 = timestep

            v, w = self.motion(timestep - t0)
            v_plot.append(v)
            w_plot.append(w)

            timestep += delta_time
            t_plot.append(timestep)

        ax1.plot(t_plot, v_plot, 'rx-')
        ax1.set_title('Linear-velocity-time graph.')
        ax2.plot(t_plot, w_plot, 'ko-')
        ax2.set_title('Angular-velocity-time graph')
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


