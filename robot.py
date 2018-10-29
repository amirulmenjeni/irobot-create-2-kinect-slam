import sys
import serial
import time
import datetime
import struct
import threading
import math
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import csv
from serial.tools import list_ports
from kinematic import Trajectory

# Opcodes
START        = chr(128)
SAFE_MODE    = chr(131)
FULL_MODE    = chr(132)
SENSORS      = chr(142)
DRIVE_DIRECT = chr(145)

# Packets
PKT_MOTION   = chr(2)  # Group Packet 2, includes Packets 17 to 20.
PKT_STATUS   = chr(3)  # Group Packet 3, includes Packets 21 to 26.
PKT_DISTANCE = chr(19)
PKT_ANGLE    = chr(20)
PKT_BATT_CHG = chr(25)
PKT_BATT_CAP = chr(26)

# Packet bytes
PKT_BYTES = { }
PKT_BYTES[PKT_MOTION]   = 6
PKT_BYTES[PKT_DISTANCE] = 2
PKT_BYTES[PKT_ANGLE]    = 2
PKT_BYTES[PKT_STATUS]   = 10

# Threads
THREAD_MOTION = 0

class Util:
    def to_twos_comp_2(val):

        """
        Returns the two's complement value corresponding to a signed integer
        value as a two-tuple. The first element of the tuple is the high byte,
        and the second element is the low byte.
        """

        if val < 0:
            val = val + (1 << 16)
        return ((val >> 8) & 0xff, val & 0xff)

    def from_twos_comp_to_signed_int(val, byte=2):

        """
        Returns the signed integer value corresponding to a two's complment
        n-byte binary (the default is 2).
        """

        range_max = int((2 ** (byte * 8)) / 2)
        ones = 2 ** (byte * 8) - 1

        if val > range_max:
            val = (val ^ ones) + 1
            return -1 * val
        return val

    def from_binary_to_unsigned_int(val):

        pass

    def msec_to_mmsec(val):
        
        """
        Simply convert m/s to mm/s.
        """

        return val * 1000.0

    def mm_to_cm(val):
        
        """
        Simply convert mm to cm.
        """

        return val * 0.1

    def cm_to_mm(val):

        """
        Simply convert cm to mm.
        """

        return val * 10.0

    def cap(val, smallest, highest):
        
        """
        Clamp a value between a smallest and a highest value (inclusive).
        """

        if val < smallest:
            return smallest
        elif val > highest:
            return highest
        return val

    def is_in_circle(center, radius, position):

        """
        Check if a point at a given position is located within a circle. 

        @param center:
            The center position of the circle. A length-2 tuple.
        @param radius:
            The radius of the circle.
        @param position:
            A length-2 tuple of the position to be tested.
        """

        d = math.sqrt(((center[0] - position[0]) ** 2) +\
                       (center[1] - position[1]) ** 2)

        return d <= radius

class PIDControler:

    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.__prev_error = 0
        self.__error = 0
        self.__setpoint = 0
        self.__integral = 0

    def e(self, sp, pv):
        self.__prev_error = self.__error
        self.__error = sp - pv

    def P(self):
        return self.kp * self.__error

    def I(self, dt):
        self.__integral = (self.__integral + self.__error * dt)
        return self.ki * self.__integral

    def D(self, dt):
        derivative = (self.__error - self.__prev_error) / dt
        return self.kd * derivative

class StaticPlotter:

    def __init__(self, n, init_pos, style='b-', on_off=None):

        self.__n = n
        self.__theta = [0] * n

        self.__pos_x = [p[0] for p in init_pos]
        self.__pos_y = [p[1] for p in init_pos]

        self.__dis_x = []
        self.__dis_y = []
        for i in range(n):
            self.__dis_x.append([self.__pos_x[i]])
            self.__dis_y.append([self.__pos_y[i]])

        self.__styles = ['b-']
        if style == 'b-':
            self.__styles = ['b-'] * n
        else:
            self.__styles = style

        if on_off is not None:
            self.__on_off = on_off
            for i in range(n):
                if i not in self.__on_off:
                    self.__on_off[i] = '111'
        else:
            self.__on_off = {}
            for i in range(n):
                self.__on_off[i] = '111'

        self.__min_x = min(self.__pos_x)
        self.__max_x = max(self.__pos_x)
        self.__min_y = min(self.__pos_y)
        self.__max_y = min(self.__pos_y)
        self.__min_v = 999
        self.__max_v = -999
        self.__min_w = 999
        self.__max_w = -999

        self.__time = [0] * n
        self.__timestep = []
        for i in range(n):
            self.__timestep.append([])
        self.__linear_velocity = []
        self.__angular_velocity = []

        for i in range(n):
            self.__linear_velocity.append([])
            self.__angular_velocity.append([])

        gs = gridspec.GridSpec(2, 2)

        self.__fig = plt.figure()

        # Subplot for velocity-time graph.
        self.__ax1 = self.__fig.add_subplot(
            gs[0, 0],
            ylim=[self.__min_v, self.__max_v])

        # Subplot for angular-velocity-time graph.
        self.__ax2 = self.__fig.add_subplot(
            gs[0, 1],
            ylim=[self.__min_w, self.__max_w])

        # Subplot for displacement.
        self.__ax3 = self.__fig.add_subplot(
            gs[1, :],
            xlim=[self.__min_x, self.__max_x],
            ylim=[self.__min_y, self.__max_y])


    def __plot(self, ax, i):
        return ax.plot(self.__pos_x[i], self.__pos_y[i],\
            self.__styles[i])[0]

    def set_waypoints(self, waypoints):
        self.__waypoints_x = [0] * len(waypoints)
        self.__waypoints_y = [0] * len(waypoints)

        for i, (x, y) in enumerate(waypoints):
            self.__waypoints_x[i] = x
            self.__waypoints_y[i] = y

    def update_plot(self, i, delta_time, delta_distance, delta_angle, v, w):

        if i < 0 or i > self.__n:
            raise ValueError('Invalid line index given.')

        delta_angle = math.radians(delta_angle)
        self.__theta[i] = self.__theta[i] + delta_angle

        self.__pos_x[i] = self.__pos_x[i] +\
            delta_distance * math.cos(self.__theta[i])
        self.__pos_y[i] = self.__pos_y[i] +\
            delta_distance * math.sin(self.__theta[i])

        self.__timestep[i].append(self.__time[i])
        self.__time[i] = self.__time[i] + delta_time

        self.__linear_velocity[i].append(v)
        self.__angular_velocity[i].append(w)

        # print('v:', len(self.__linear_velocity[i]))
        # print('t:', len(self.__timestep[i]))

        if i in self.__on_off and self.__on_off[i][2] == '1':
            if self.__pos_x[i] < self.__min_x:
                self.__min_x = self.__pos_x[i]
            if self.__pos_x[i] > self.__max_x:
                self.__max_x = self.__pos_x[i]
            if self.__pos_y[i] < self.__min_y:
                self.__min_y = self.__pos_y[i]
            if self.__pos_y[i] > self.__max_y:
                self.__max_y = self.__pos_y[i]

        if i in self.__on_off and self.__on_off[i][0] == '1':
            if v < self.__min_v:
                self.__min_v = v
            if v > self.__max_v:
                self.__max_v = v

        if i in self.__on_off and self.__on_off[i][1] == '1':
            if w < self.__min_w:
                self.__min_w = w
            if w > self.__max_w:
                self.__max_w = w

        self.__dis_x[i].append(self.__pos_x[i])
        self.__dis_y[i].append(self.__pos_y[i])

    def draw(self):

        v_margin = 0.05 * (self.__max_v - self.__min_v)
        w_margin = 0.05 * (self.__max_w - self.__min_w)
        x_margin = 0.05 * (self.__max_x - self.__min_x)
        y_margin = 0.05 * (self.__max_y - self.__min_y)

        self.__ax1.set_ylim(ymin=self.__min_v - v_margin, 
                            ymax=self.__max_v + v_margin)
        self.__ax2.set_ylim(ymin=self.__min_w - w_margin,
                            ymax=self.__max_w + w_margin)
        self.__ax3.set_xlim(xmin=self.__min_x - x_margin,
                            xmax=self.__max_x + x_margin)
        self.__ax3.set_ylim(ymin=self.__min_y - y_margin, 
                            ymax=self.__max_y + y_margin)

        self.__ax1.set_xlabel('Time')
        self.__ax1.set_ylabel('Linear velocity')

        self.__ax2.set_xlabel('Time')
        self.__ax2.set_ylabel('Angular velocity')

        self.__ax3.set_xlabel('X-displacement')
        self.__ax3.set_ylabel('Y-displacement')

        for i in range(self.__n):
            if i in self.__on_off:
                if self.__on_off[i][0] == '1':
                    self.__ax1.plot(self.__timestep[i], 
                        self.__linear_velocity[i], self.__styles[i])
                if self.__on_off[i][1] == '1':
                    self.__ax2.plot(self.__timestep[i],
                        self.__angular_velocity[i], self.__styles[i])
                if self.__on_off[i][2] == '1':
                    self.__ax3.plot(self.__dis_x[i], self.__dis_y[i],
                        self.__styles[i])

        # Waypoints.
        if self.__waypoints_x is not None and\
           self.__waypoints_y is not None:
            self.__ax3.plot(self.__waypoints_x, self.__waypoints_y, 'bo-')

        plt.grid()
        plt.show()

class Robot:

    def __init__(self, b=26, port='', max_speed=10,
        pid_v=(0, 0, 0), pid_w=(0, 0, 0)):

        """
        Initialize the Robot class. 
        
        port: The serial port to connect with the iRobot Create e.g.
        '/dev/ttyUSB0'.

        @param b:
            The axial distance between the wheels in cm.
        @param port:
            The serial port used to send messages to the robot.
        @param max_speed:
            The maximum speed each wheel can attain in cm/s.
        """

        # Serial variables.
        self.baudrate=57600

        # If port is not explicitly given, do an automatic port look-up of
        # a USB serial port.
        if port == '':
            try:
                port = self.ports_lookup()[0]
                print('Found port:', port)
            except:
                sys.exit('ERROR: No USB serial port found.')

        self.__b = b
        self.__port = port
        self.ser = serial.Serial(port, baudrate=self.baudrate, timeout=1)

        # Store issued command variables.
        self.issued_v = 0
        self.issued_w = 0

        # Reading of linear and angular velocity of the robot chassis.
        self.__v = 0
        self.__w = 0
        self.__delta_distance = 0
        self.__delta_angle = 0

        # Kinematics variables.
        self.max_speed = max_speed # in cm/s

        # The position and orientation of the robot w.r.t global reference
        # frame.
        self.__pose = (0, 0, 0)

        # PID controllers.
        # self.pid_v = PIDControler(1.5, 0.015, 0.0)
        # self.pid_w = PIDControler(0.65, 0.05, 0.0)
        self.pid_v = PIDControler(pid_v[0], pid_v[1], pid_v[2])
        self.pid_w = PIDControler(pid_w[0], pid_w[1], pid_w[2])

        # Flags.
        self.is_pid_enable = True
        self.is_autonomous = False

        # Autonomous driving variables.
        self.auto_trajectory = None
        self.auto_timestep = 0
        self.auto_end_time = 0
        self.auto_t0 = 0

        self.plotter = StaticPlotter(
            3, # Plot 3 lines for each graph.
            [self.get_pose()[:2]] * 3, 
            ['kx-', 'cx--', 'g^--'],
            {0: '111',
             1: '111',
             2: '000'}) # Plot number 2 is not drawn

        self.__is_timestep_end = False

        # Initialize all threads.
        self.init_threads()

    def ports_lookup(self):

        """
        Automatic lookup of the roomba serial port device. If no USB serial port
        device is found, return False. Otherwise, the USB serial port(s) found
        will be returned in a list.
        """

        roomba_ports = [
            p.device
            for p in list_ports.comports()
        ]

        if len(roomba_ports) == 0:
            raise

        return roomba_ports

    def init_threads(self):

        """
        Initialize all threads.
        """

        self.threads = [
                threading.Thread(target=Robot.thread_motion, args=(self,),\
                name='Motion'),\
        ]

        # A list of flag to maintain thread stop request
        # via stop_thread() function.
        self.is_thread_stop_requested = [False] * len(self.threads)

        for thread in self.threads:
            if thread.is_alive():
                print('Thread %s has already started.' % thread.name)
            else:
                thread.start()
                print('Thread "%s" started' % thread.name)

    def clean_up(self):
        self.stop_all_threads()

    def start(self):

        """
        Start the iRobot Create.
        """

        self.send_code(START)

    def safe_mode(self):
        
        """
        Set the robot to safe mode.
        """

        self.send_code(SAFE_MODE)

    def full_mode(self):

        """
        Set the robot to full mode.
        """

        self.send_code(FULL_MODE)

    def send_msg(self, opcode, byte_data):
        
        """
        Deprecated. I think using send_codes is sufficient. For now.
        """

#         print('message:', opcode + byte_data)
#         print('len    :', len(opcode + byte_data))
        successful = False
        while not successful:
            try:
                self.send_code(opcode + byte_data)
                successful = True
                print('Success!')
            except serial.SerialException as e:
                print('Error:', e)

    def send_code(self, code):

        """
        Send a byte to the iRobot Create. To send a code, say, 128, to start
        the iRobot, you may pass chr(128).
        """

        try:
            self.ser.write(bytes(code, encoding='Latin-1'))
        except serial.SerialException as e:
            print('Error: send_code(code) error') 
            
    def send_codes(self, codes):

        """
        Send a list of 1 byte binary datum to the iRobot Create.
        See send_code.
        """

        for c in codes:
           self.send_code(c)

    def recv_code(self, packet_id):
        
        """
        Read from sensor. Pass the packet_id to get the buffer returned from
        requesting the sensor reading. The packet_id is a 1 byte binary.
        """

        codes = [SENSORS, packet_id]
        self.send_codes(codes)
        read_buf = self.ser.read(PKT_BYTES[packet_id])

        return read_buf

    def interpret_code(self, packet_id, read_buf):

        """
        Interpret a buffer. Refer the manual on how the buffer for each sensor
        packet buffer should be interpreted.
        """

        try:
            if packet_id == PKT_MOTION:

                distance = struct.unpack('>i',
                        b'\x00\x00' + read_buf[2:4])[0]
                angle = struct.unpack('>i',
                        b'\x00\x00' + read_buf[4:6])[0]

                # Distance in mm, angle in degrees.
                distance = Util.from_twos_comp_to_signed_int(distance, byte=2)
                angle = Util.from_twos_comp_to_signed_int(angle, byte=2)

                distance = Util.mm_to_cm(distance)

                return distance, angle

            if packet_id == PKT_STATUS:

                charge = struct.unpack('>i',
                        b'\x00\x00' + read_buf[7:9])[0]
                capacity = struct.unpack('>i',
                        b'\x00\x00' + read_buf[9:11])[0]

                print('charge:', charge)
                print('capacity:', capacity)

                charge = int(charge, 2)
                capacity = int(capacity, 2)

                return charge, capacity

            if packet_id == PKT_DISTANCE:
                
                # The buffer received from PKT_DISTANCE is 2 bytes, but
                # struct.unpack requires 4 bytes to convert it to integer
                # (hence it's prepended with b'\x00\x00').
                # Convert the 2 bytes binary data to integer data. This integer
                # data represents distance in mm.
                #
                # ">i" means the buffer is read as signed int, big endian.
                i = struct.unpack('>i', b'\x00\x00' + read_buf)[0]

                # Maximum of the integer value replied for this packet.
                # Since i is a signed integer, we perform two's complement
                # and get its actual value.
                i = Util.from_twos_comp_to_signed_int(i, byte=2)

                # Convert from mm/s to cm/s
                return Util.mm_to_cm(i)

            if packet_id == PKT_ANGLE:
                i = struct.unpack('>i', b'\x00\x00' + read_buf)[0]
                i = Util.from_twos_comp_to_signed_int(i, byte=2)
                return i

            if packet_id == PKT_BATT_CHG:
                # ">I" means the buffer is read as unsigned int, big endian.
                i = struct.unpack('>I', b'\x00\x00' + read_buf)[0]
                return i

            if packet_id == PKT_BATT_CAP:
                i = struct.unpack('>I', b'\x00\x00' + read_buf)[0]
                return i

            else:
                pass

        except struct.error as e:
            pass
            # print(e)
            # print('read_buf:', read_buf)
            # print('length:', len(read_buf))

        return 0.0

    def drive_direct(self, lw, rw):
    
        """
        Drive each left wheel (lw) and right wheel (rw) in cm/s.
        """

        # Convert variables from cm/s to mm/s.
        max_speed = Util.cm_to_mm(self.max_speed)
        lw = Util.cm_to_mm(lw)
        rw = Util.cm_to_mm(rw)

        # Cap linear speed of each wheel.
        lw = Util.cap(lw, -max_speed, +max_speed)
        rw = Util.cap(rw, -max_speed, +max_speed)

        rw_high, rw_low = Util.to_twos_comp_2(int(rw))
        lw_high, lw_low = Util.to_twos_comp_2(int(lw))

        codes = [START, FULL_MODE,
                 DRIVE_DIRECT,
                 chr(rw_high), chr(rw_low),
                 chr(lw_high), chr(lw_low)]
            
        self.send_codes(codes)

    def drive(self, vel_forward, vel_angular, is_feedback=False):

        """
        Drive the robot given its local forward velocity and its angular
        velocity (in radian per seconds.).

        NOTE: This only update the issued_v and issued_w of the robot. The
        actual command is issued in thread_motion().
        """
        
        if not is_feedback:
            self.issued_v = vel_forward
            self.issued_w = vel_angular 

        # v1 = vel_forward - self.__b * vel_angular
        # v2 = vel_forward + self.__b * vel_angular
        # self.drive_direct(v1, v2)

    def drive_to(self, speed, next_pos, targ_orientation=None):

        """
        This method is the same as drive_trajectory, except that this takes in
        only the next waypoint instead of a series of waypoints.
        """

        print('Drive from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f).' %\
                (self.__pose[0], self.__pose[1], self.__pose[2],
                 targ_pos[0], next_pos[1], targ_orientation))

        self.auto_trajectory = Trajectory(speed, 
            self.__pose[:2], self.__pose[2],
            next_pos, targ_orientation)

        print('Start autonomous driving.')
        self.is_autonomous = True
        self.auto_timestep = 0
        self.auto_t0 = 0

    def drive_trajectory(self, speed, waypoints, targ_orientation=None):

        """
        Drive the robot at a set speed through a set of waypoints. By default,
        the orientation of the robot is automatically calculated from the last
        two waypoints. The speed will is used to determine the magnitude of the
        set of velocities at each point of the calculated trajectory produced by
        interpolation of the robot's velocity at each waypoint.
        """

        self.auto_trajectory = Trajectory(speed,
            self.__pose[:2], self.__pose[2],
            waypoints[0], targ_orientation)

        if len(waypoints) > 1:
            for i in range(1, len(waypoints)):
                x, y = waypoints[i]
                self.auto_trajectory.add_waypoint(x, y)

        time_estimate =\
            self.auto_trajectory.estimate_time_between_points(
                self.auto_trajectory.get_speed(),
                self.auto_trajectory.current())
        self.auto_end_time = time_estimate

        self.is_autonomous = True
        self.auto_timestep = 0
        self.auto_t0 = 0
        self.auto_speed = speed

    def stop_thread(self, i):
        
        """
        Stop the i-th thread in the self.threads list. The robot object maintain
        a list of flag corresponding the each thread in self.threads list called
        is_thread_stop_requested. When is_thread_stop_requested[i] is set to
        True, the loop that runs in that thread is stopped, stopping the thread.
        Note that the function that the thread run is responsible to set the
        flag back to false.
        """

        self.is_thread_stop_requested[i] = True

    def stop_all_threads(self):

        """
        Simply stop all threads. See stop_thread.
        """

        for i in range(0, len(self.threads)):
            self.stop_thread(i)

    def show_running_threads(self):

        """
        Return a list of thread by their names. Only threads that is running
        (i.e. thread.is_live() is True) is included in the list.
        """

        running_threads = []
        for t in self.threads:
            if t.is_alive():
                running_threads.append(t.name)
        return running_threads

    def start_thread(self, i):

        """
        Start the i-th thread from self.threads list.
        """

        if not self.threads[i].is_alive():
            self.threads[i].start()

    def thread_motion(self):

        """
        Keep reading the encoder values on the robot's pose.
        """

        delta_time = 1
        prev_traj_v = None
        prev_traj_w = None

        while True:

            # Read distance and angle.
            delta_distance, delta_angle = 0, 0
            try:
                self.get_sensor(PKT_MOTION)
            except:
                pass
            time.sleep(delta_time)
            try:
                delta_distance, delta_angle = self.get_sensor(PKT_MOTION)
            except:
                pass

            # Compute the linear and angular velocity from measured distance and
            # angle within delta_time respectively.
            read_v = delta_distance / delta_time # Forward velocity
            read_w = delta_angle / delta_time    # Change in orientation (degree)

            self.__delta_distance = delta_distance
            self.__delta_angle = delta_angle

            # Update the position of the robot.
            self.__update_position(delta_distance, delta_angle)

            v1, v2 = 0, 0

            if self.is_autonomous:
                # Autonomous driving.

                # If the time duration estimated to reach the next waypoint in
                # the trajectory is T, then every T/4 seconds:
                #  1. If the robot has not reach the next estimated position in
                #     the trajectory, recalculate the trajectory.
                #  2. Otherwise, if we have reached the next estimated position
                #     in the trajectory, we have two possibilities: either we're
                #     at the final waypoint, or we're at the intermediate
                #     waypoint. When we're at the final waypoint, simply
                #     stop driving. Otherwise, we estimate the time to reach the
                #     next waypoint.
                if self.auto_timestep >= self.auto_end_time / 4.0:

                    next_pos = self.auto_trajectory.displacement(
                        self.auto_end_time)

                    if not Util.is_in_circle(next_pos, 15.0,
                        self.__pose[:2]):

                        self.is_autonomous = False
                        self.auto_timestep = 0
                        waypoints = self.auto_trajectory.get_waypoints()[\
                                self.auto_trajectory.current() + 1:]

                        self.drive_trajectory(self.auto_speed, waypoints)

                    else:
                        self.test_song()
                        self.auto_trajectory.next()

                        if self.auto_trajectory.is_final_waypoint():
                            self.is_autonomous = False
                            self.auto_timestep = 0
                            v1, v2 = 0, 0
                        else:
                            time_estimate =\
                                self.auto_trajectory.estimate_time_between_points(
                                    self.auto_trajectory.get_speed(),
                                    self.auto_trajectory.current())
                            self.auto_end_time = self.auto_end_time + time_estimate 
                            self.auto_t0 = self.auto_timestep

                # If the current waypoint (i.e., the latest reached waypoint)
                # is not the final waypoint, then continue to drive
                # autonomously.
                if not self.auto_trajectory.is_final_waypoint():

                    input_v, input_w = self.auto_trajectory.motion(
                        self.auto_timestep - self.auto_t0)

                    v1, v2 = Robot.__inverse_drive(
                        input_v, input_w, self.__b)

                    # Update sensor reading plot.
                    self.plotter.update_plot(0, delta_time, 
                        delta_distance, delta_angle,
                        read_v, read_w)

                    # Update trajectory calculation plot.
                    self.plotter.update_plot(1, delta_time, 
                        input_v * delta_time,
                        math.degrees(input_w) * delta_time,
                        input_v, math.degrees(input_w))

                    # self.plotter.update_plot(2, delta_time, 
                    #     controlled_v * delta_time, math.degrees(w) * delta_time,
                    #     controlled_v, math.degrees(controlled_w))

                    self.auto_timestep = self.auto_timestep + delta_time

                self.drive_direct(v1, v2)

            else:
                # Manual driving.
                v1, v2 = Robot.__inverse_drive(
                    self.issued_v, self.issued_w, self.__b)
                self.drive_direct(v1, v2)

            if self.is_thread_stop_requested[THREAD_MOTION]:
                break

    def wait_for_next_timestep(self):
        while not self.__is_timestep_end:
            time.sleep(0.01)

    def halt(self):

        """
        Abruptly stop the robot from moving. This simply uses the drive_direct
        function and setting each wheel's linear speed to zero.
        """

        self.drive(0, 0)
        self.is_autonomous = False
        self.auto_timestep = 0

    def test_song(self):

        """
        Test the iRobot Create by making it sing a song.
        """

        codes = [128, 132, 140, 0, 4, 62, 12, 66, 12, 66, 12, 69, 12, 74,
                 36, 141, 0]
        self.send_codes([chr(c) for c in codes])

    def test_poll_distance(self, cms=1, rate=1, reverse=False):
        if reverse:
            cms *= -1
        self.drive_direct(cms, cms)
        while True:
            t = time.time()
            delta_distance = self.get_sensor(PKT_DISTANCE)
            t = time.time() - t
            t = Util.cap(t, 0, t + 0.5)

            print('delta distance:', delta_distance)
            print('speed:', float(float(delta_distance) / t))

    def test_poll_angle(self, cms=1, rate=1):
        self.drive_direct(cms, 0)
        total_angle = 0
        while True:
            t = time.time()
            delta_angle = self.get_sensor(PKT_ANGLE)
            t = time.time() - t
            t = Util.cap(t, 0, t + 0.5)
            print('delta angle:', delta_angle)
            total_angle += delta_angle
            print('total angle:', total_angle)

            print('angular speed:', float(float(delta_angle) / t))
            
            delta_distance = self.get_sensor(PKT_DISTANCE)
            print('delta distance:', delta_distance)
            time.sleep(rate)

    def is_serial_open(self):

        """
        Returns true if the serial through which this program is connected to
        the robot is open. Otherwise, this function returns false.
        """

        return self.ser.is_open

    def reconnect(self):
    
        """
        Reconnect the serial.
        """
            
        self.ser.close()
        self.ser = serial.Serial(port, self.baudrate, timeout=1)
        self.ser = Serial(port, self.baudrate, timeout=1)

    def battery_charge(self):

        """
        Returns the value of the current battery percentage.
        """

        charge, capacity = self.get_sensor(PKT_STATUS)
        return float(charge / capacity)

    def get_sensor(self, packet_id):

        """
        Get the interpreted value of a given sensor indicated by packet_id.
        packet_id is a 1 byte binary. Refer to the manual for the packet id of a
        given sensor.
        """

        return self.interpret_code(packet_id, self.recv_code(packet_id))

    def get_motion(self):

        """
        Returns the linear and angular velocity of the robot.
        """

        return (self.__v, self.__w)

    def get_pose(self):

        """
        Returns the position of the robot in cartesian coordinate of in the
        inertial reference frame.
        """

        return self.__pose

    def reset_pose(self):

        self.__pose = (0, 0, 0)

    def get_delta_distace(self):
        return self.__delta_distance

    def get_delta_angle(self):
        return self.__delta_angle

    def enable_pid(self):
        self.is_pid_enable = True
        
    def disable_pid(self):
        self.is_pid_enable = False

    def __update_position(self, delta_distance, delta_angle):

        """
        Updates the cartesian coordinate position of the robot chassis in the
        inertial reference frame. This assumes the delta distance and delta
        angle given is taken at a consistent time ticks.

        Parameters:
            delta_distance: 
                The distance traveled within a consistent time tick.
            delta_angle:
                The change in angle within a consistent time tick.
        """

        # Update orientation.
        orientation = (self.__pose[2] + delta_angle) % 360

        # Convert the orientation from degree to radian for the next operations
        # (determining the x- and y-positon).
        radian = (math.pi / 180.0) * orientation

        # Update x- and y-positon.
        x = self.__pose[0] + delta_distance * math.cos(radian)
        y = self.__pose[1] + delta_distance * math.sin(radian)

        self.__pose = (x, y, orientation)

    def __inverse_drive(v, w, b):

        return v - b * w, v + b * w

    def __forward_drive(v1, v2, b):

        return ((v1 + v2) / 2.0), (v1 - v2) / (2.0 * b)
