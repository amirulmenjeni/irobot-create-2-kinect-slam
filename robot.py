import serial
import time
import datetime
import struct
import threading
import numpy as np

# Opcodes
START        = chr(128)
SAFE_MODE    = chr(131)
FULL_MODE    = chr(132)
SENSORS      = chr(142)
DRIVE_DIRECT = chr(145)

# Packets
PKT_DISTANCE = chr(19)
PKT_ANGLE    = chr(20)
PKT_BATT_CHG = chr(25)
PKT_BATT_CAP = chr(26)

# Threads
THREAD_MOTION = 0

class Util:
    def to_twos_comp_2(val):
        if val < 0:
            val = val + (1 << 16)
        return ( (val >> 8) & 0xff, val & 0xff)

    def from_twos_comp_to_signed_int(val, byte=2):
        range_max = int((2 ** (byte * 8)) / 2)
        ones = 2 ** (byte * 8) - 1

        if val > range_max:
            val = (val ^ ones) + 1
            return -1 * val
        return val

    def msec_to_mmsec(val):
        return val * 1000.0

    def cmsec_to_mmsec(val):
        return val * 10

    def cap(val, smallest, highest):
        if val < smallest:
            return smallest
        elif val > highest:
            return highest
        return val

class Kinematics:

    def __init__(self, robot):
        self.theta = 0

        """
        Initialize the kinematics of this robot and consequently establish its
        intertial frame.
        """

        self.robot = robot

class Robot:

    def __init__(self, port, max_speed=10):

        # Serial variables.
        self.baudrate=57600
        self.port = port
        self.ser = serial.Serial(port, baudrate=self.baudrate, timeout=1)

        # Kinematics variables.
        self.instant_speed = 0
        self.instant_ang_vel = 0
        self.max_speed = max_speed # in cm/s
        self.motion = np.zeros((3, 1))

        print('Time initialized:', datetime.datetime.now())
        print('Connecting via serial port %s with %s baud.'\
               % (port, self.baudrate))

        # Initialize all threads.
        print('Initializing threads...')
        self.init_threads()

    def init_threads(self):
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

    def __init__(self, port):
        self.baudrate=57600
        self.port = port
        self.ser = serial.Serial(port, baudrate=self.baudrate, timeout=1)
        self.speed = 0
        self.motion = np.zeros((3, 1))

        print('Time initialized:', datetime.datetime.now())

        print('Connecting via serial port %s with %s baud.'\
               % (port, self.baudrate))

    def start(self):
        self.send_code(START)

    def safe_mode(self):
        self.send_code(SAFE_MODE)

    def full_mode(self):
        self.send_code(FULL_MODE)

    def send_msg(self, opcode, byte_data):
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
        try:
            self.ser.write(bytes(code, encoding='Latin-1'))
        except serial.SerialException as e:
            print('Error: send_code(code) error') 
            
    def send_codes(self, codes):
        for c in codes:
           self.send_code(c)


    def recv_code(self, packet_id):
        codes = [SENSORS, packet_id]
        self.send_codes(codes)
        read_buf = self.ser.read(4)

        return read_buf

    def interpret_code(self, packet_id, read_buf):

        try:
            if packet_id == PKT_DISTANCE:
                
                # The buffer received from PKT_DISTANCE is 2 bytes, but
                # struct.unpack requires 4 bytes to convert it to float.
                # Convert the 2 bytes binary data to integer data. This integer
                # data represents distance in mm.
                i = struct.unpack('>i', b'\x00\x00' + read_buf)[0]

                # Maximum of the integer value replied for this packet.
                # Since i is a signed integer, we perform two's complement
                # and get its actual value.
                i = Util.from_twos_comp_to_signed_int(i, byte=2)

                # Convert from mm/s to cm/s
                return i / 10.0

            if packet_id == PKT_ANGLE:
                i = struct.unpack('>i', b'\x00\x00' + read_buf)[0]
                i = Util.from_twos_comp_to_signed_int(i, byte=2)
                return i

            if packet_id == PKT_BATT_CHG:
                i = struct.unpack('>I', b'\x00\x00' + read_buf)[0]
                return i

            if packet_id == PKT_BATT_CAP:
                i = struct.unpack('>I', b'\x00\x00' + read_buf)[0]
                return i

            else:
                pass

        except struct.error as e:
            print(e)
            print('read_buf:', read_buf)
            print('length:', len(read_buf))

        return 0.0

    def drive_direct(self, lw, rw):
    
        """
        Drive each left wheel (lw) and right wheel (rw) in cm/s.
        """

        # Convert variables from cm/s to mm/s.
        lw = Util.cmsec_to_mmsec(lw)
        rw = Util.cmsec_to_mmsec(rw)
        max_speed = Util.cmsec_to_mmsec(self.max_speed)

            
        # Cap linear speed of each wheel.
        lw = Util.cap(lw, -max_speed, +max_speed)
        rw = Util.cap(rw, -max_speed, +max_speed)

        lw = Util.cmsec_to_mmsec(lw)
        rw = Util.cmsec_to_mmsec(rw)

        rw_high, rw_low = Util.to_twos_comp_2(int(rw))
        lw_high, lw_low = Util.to_twos_comp_2(int(lw))

        codes = [START, FULL_MODE,
                 DRIVE_DIRECT,
                 chr(rw_high), chr(rw_low),
                 chr(lw_high), chr(lw_low)]
            
        self.send_codes(codes)

    def stop_thread(self, i):
        self.is_thread_stop_requested[i] = True

    def stop_all_threads(self):
        for i in range(0, len(self.threads)):
            self.stop_thread(i)

    def show_running_threads(self):
        running_threads = []
        for t in self.threads:
            if t.is_alive():
                running_threads.append(t.name)
        return running_threads

    def start_thread(self, i):
        if not self.threads[i].is_alive():
            self.threads[i].start()

    def poll_motion(self):

        """
        Updates the Robot variables that defines its motion: instant_speed, 
        instant_ang_vel
        """
        n = 2 # The number of motion variables.

        while True:
            delta_time = time.time()
            delta_distance = self.get_sensor(PKT_DISTANCE)
            delta_time = time.time() - delta_time
            self.instant_speed = float(delta_distance / (delta_time * n))

            delta_time = time.time()
            delta_angle = self.get_sensor(PKT_ANGLE)
            delta_time = time.time() - delta_time
            self.instant_ang_vel = float(delta_angle / (delta_time * n))

            if self.is_thread_stop_requested[THREAD_MOTION]:
                break

    def halt(self):
        self.drive_direct(0, 0)
    
    def test_song(self):
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
        return self.ser.is_open

    def reconnect(self):
        self.ser.close()
        self.ser = serial.Serial(port, self.baudrate, timeout=1)
        self.ser = Serial(port, self.baudrate, timeout=1)

    def battery_charge(self):
        current  = self.get_sensor(PKT_BATT_CHG)
        capacity = self.get_sensor(PKT_BATT_CAP)
        return float(current / capacity)

    def get_sensor(self, packet_id):
        return self.interpret_code(packet_id, self.recv_code(packet_id))


