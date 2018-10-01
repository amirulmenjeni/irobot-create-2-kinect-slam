import sys
import serial
import time
import datetime
import struct
import threading
import math
import numpy as np
from serial.tools import list_ports

# Opcodes
START        = chr(128)
SAFE_MODE    = chr(131)
FULL_MODE    = chr(132)
SENSORS      = chr(142)
DRIVE_DIRECT = chr(145)

# Packets
PKT_MOTION   = chr(2)  # Group Packet 2, includes Packets 17 to 20.
PKT_DISTANCE = chr(19)
PKT_ANGLE    = chr(20)
PKT_BATT_CHG = chr(25)
PKT_BATT_CAP = chr(26)

# Packet bytes
PKT_BYTES = { }
PKT_BYTES[PKT_MOTION]    = 6
PKT_BYTES[PKT_DISTANCE] = 2
PKT_BYTES[PKT_ANGLE]    = 2

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

class Kinematics:

    def __init__(self, b):

        """
        Initialize the kinematics of this robot and consequently establish its
        intertial frame.

        Parameters:
            b: The axial distance between the wheels of the differential drive
               mobile robot.
        """
        
        self.__b = b
        self.__orientation = 0
        self.__x_position = 0
        self.__y_position = 0

    def inverse(self, targ_v, targ_a):

        """
        Returns the required setpoint speed for each individual wheel to achieve
        the target speed and the target rotational speed of robot. Returns a
        2-tuple: indexed 0 and 1 for the speed of the left and right wheel
        respectively.

        Parameters:
            targ_v: The desired velocity of the chassis of the robot.
            targ_a: The desired change in orientation of the chassis of the
                    robot.

        v_1 = v - b * theta_dot
        v_2 = v + b * theta_dot
        """

        return (targ_v - self.__b * targ_a, targ_v + self.__b * targ_a)

    def forward(self, v1, v2):
        
        """
        Returns the setpoint motion given the issued linear wheel velocity v1
        and v2 of the left and right wheel respectively.

        Parameters:
            v1: The linear velocity of the left wheel local to the left wheel's
                frame.
            v2: The linear velocity of the right wheel local to the right
                wheel's frame.
        """

        v = (v1 + v2) / 2
        w = (v1 - v2) / (2 * self.__b)

        return v, w

    def update_cartesian_position(self, delta_distance, delta_angle):

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

        self.__orientation = (self.__orientation + delta_angle) % 360

        radian = (math.pi / 180.0) * (self.__orientation + delta_angle)

        self.__x_position += delta_distance * math.cos(radian)
        self.__y_position +=  delta_distance * math.sin(radian)

    def get_position(self):

        """
        Returns the position of the robot in the inertial reference frame in
        cartesian coordinate form.
        """
        return (self.__x_position, self.__y_position)

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
        self.__integral = self.ki * (self.__integral + self.__error * dt)
        return self.__integral

    def D(self, dt):
        derivative = (self.__error - self.__prev_error) / dt
        return self.kd * derivative

class Robot:

    def __init__(self, port='', max_speed=10):

        """
        Initialize the Robot class. 
        
        port: The serial port to connect with the iRobot Create e.g.
        '/dev/ttyUSB0'.

        max_speed: The maximum speed each wheel can attain in cm/s.
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

        self.port = port
        self.ser = serial.Serial(port, baudrate=self.baudrate, timeout=1)

        # Store issued command variables.
        self.issued_v = 0
        self.issued_w = 0

        # Reading of linear and angular velocity of the robot chassis.
        self.__v = 0
        self.__w = 0

        # Kinematics variables.
        self.max_speed = max_speed # in cm/s
        self.kinematics = Kinematics(5)

        # PID controllers.
        self.pid_v = PIDControler(0.00, 0.01, 0.01)
        self.pid_w = PIDControler(0.01, 0.01, 0.01)

        # Flags.
        self.is_pid_enable = True
        self.is_drive = False

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
                threading.Thread(target=Robot.poll_motion, args=(self,),\
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

            if packet_id == PKT_DISTANCE:
                
                # The buffer received from PKT_DISTANCE is 2 bytes, but
                # struct.unpack requires 4 bytes to convert it to float
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
        """
        
        if not is_feedback:
            self.issued_v = vel_forward
            self.issued_w = vel_angular 
            self.is_drive = True

        v1, v2 = self.kinematics.inverse(vel_forward, vel_angular)
        self.drive_direct(v1, v2)

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

    def poll_motion(self):

        """
        Updates the Robot variables that defines its motion.
        """

        delta_time = 0.5

        while True:

            self.get_sensor(PKT_MOTION)
            time.sleep(delta_time)
            delta_distance, delta_angle = self.get_sensor(PKT_MOTION)

            v = delta_distance / delta_time # Forward velocity
            w = delta_angle / delta_time    # Change in orientation

            self.__v = v
            self.__w = w

            self.kinematics.update_cartesian_position(\
                    delta_distance, delta_angle)

            # if self.is_drive:
            #     print('->', (v, w), self.kinematics.get_position())

            if self.is_thread_stop_requested[THREAD_MOTION]:
                break

    def halt(self):

        """
        Abruptly stop the robot from moving. This simply uses the drive_direct
        function and setting each wheel's linear speed to zero.
        """

        self.drive(0, 0)
        self.is_drive = False

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

        current  = self.get_sensor(PKT_BATT_CHG)

        # Avoid division by zero error. 
        timeout = time.time()
        while capacity == 0.0 and timeout < 5.0:
            capacity = self.get_sensor(PKT_BATT_CAP)
            timeout = time.time() - timeout
            time.sleep(0.2)

        return float(current / capacity)

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

    def get_position(self):

        """
        Returns the position of the robot in cartesian coordinate of in the
        inertial reference frame.
        """

        return self.kinematics.get_position()

    def enable_pid(self):
        self.is_pid_enable = True
        
    def disable_pid(self):
        self.is_pid_enable = False


