import serial
import time
import datetime
import struct
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

class Kinematics:

    def __init__(self, robot):
        self.theta = 0

        """
        Initialize the kinematics of this robot and consequently establish its
        intertial frame.
        """

        self.robot = robot

class Robot:

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

#         print('read_buf len:', len(read_buf))
#         print('received msg:', read_buf)

        return read_buf

    def interpret_code(self, packet_id, read_buf):

        print('Interpret:', read_buf)

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
                i = struct.unpack('>i', b'\x00\x00' + read_buf)
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
        Drive each left wheel (lw) and right wheel (rw) in m/s.
        """

        lw = Util.cmsec_to_mmsec(lw)
        rw = Util.cmsec_to_mmsec(rw)

#         print('LW: %s cm/s' % lw)
#         print('RW: %s cm/s' % rw)
            
        # Cap at (-50, 50) cm/s
        if lw < -50:
            lw = -50
        if lw > 50:
            lw = 50
        if rw < -50:
            rw = -50
        if rw > 50:
            rw = 50

        rw_high, rw_low = Util.to_twos_comp_2(int(rw))
        lw_high, lw_low = Util.to_twos_comp_2(int(lw))

#         print('v:', rw_high, rw_low, lw_high, lw_low)

        codes = [START, FULL_MODE,
                 DRIVE_DIRECT,
                 chr(rw_high), chr(rw_low),
                 chr(lw_high), chr(lw_low)]
            
        self.send_codes(codes)

    def poll_distance_thread(rate=1):
        pass

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
            delta_distance = self.get_sensor(PKT_DISTANCE)
            print('delta distance:', delta_distance)
            time.sleep(rate)

    def is_serial_open(self):
        return self.ser.is_open

    def reconnect(self):
        self.ser.close()
        self.ser = Serial(port, self.baudrate, timeout=1)

    def battery_charge(self):
        current  = self.get_sensor(PKT_BATT_CHG)
        capacity = self.get_sensor(PKT_BATT_CAP)
        return float(current / capacity)

    def get_sensor(self, packet_id):
        return self.interpret_code(packet_id, self.recv_code(packet_id))


