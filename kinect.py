import sys
import numpy as np
import cv2
from openni import openni2, nite2, utils
from openni import _openni2, _nite2

class Kinect:

    # Taken from _nite2.py.
    JOINT_HEAD = 0
    JOINT_NECK = 1
    JOINT_LEFT_SHOULDER = 2
    JOINT_RIGHT_SHOULDER = 3
    JOINT_LEFT_ELBOW = 4
    JOINT_RIGHT_ELBOW = 5
    JOINT_LEFT_HAND = 6
    JOINT_RIGHT_HAND = 7
    JOINT_TORSO = 8
    JOINT_LEFT_HIP = 9
    JOINT_RIGHT_HIP = 10
    JOINT_LEFT_KNEE = 11
    JOINT_RIGHT_KNEE = 12
    JOINT_LEFT_FOOT = 13
    JOINT_RIGHT_FOOT = 14

    def __init__(self, redist):

        openni2.initialize(redist)
        nite2.initialize(redist)
        
        self.dev = openni2.Device.open_any()

        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream = self.dev.create_color_stream()

        self.depth_stream.set_mirroring_enabled(False)
        self.color_stream.set_mirroring_enabled(False)

        self.depth_stream.start()
        self.color_stream.start()

        try:
            self.user_tracker = nite2.UserTracker(self.dev)
        except utils.NiteError:
            print('Unable to start the NiTE human tracker. '
                  'Check the error messages in the console. Model data '
                  '(s.dat, h.dat...) might be inaccessible.')
            sys.exit(-1)

        # This causes error.
        # self.dev.set_depth_color_sync_enabled(True)

        # Align depth to rgb.
        self.dev.set_image_registration_mode(\
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    def get_rgb(self):

        data = self.color_stream.read_frame().get_buffer_as_uint8()
        bgr = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def get_depth(self):

        data = self.depth_stream.read_frame().get_buffer_as_uint16()
        dmap = np.frombuffer(data, dtype=np.uint16).reshape(480, 640)
        return dmap
    
    def get_users(self):

        frame = self.user_tracker.read_frame()

        users = []

        if frame.users:
            for user in frame.users:
                if user.is_new():
                    self.user_tracker.start_skeleton_tracking(user.id)
                users.append(user)

        return users

    def depth_display(depth_map):

        d = np.uint8(depth_map.astype(float) * 255 / 2**12 - 1)
        d = 255 - cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)
        return d

    def cleanup(self):
        self.color_stream.stop()
        self.depth_stream.stop()
        openni2.unload()
        nite2.unload()

if __name__ == '__main__':

    kin = Kinect('./redist')

    while 1:
        rgb = kin.get_rgb()

        dmap = kin.get_depth()
        dimg = Kinect.depth_display(dmap)

        for user in kin.get_users():
            print('id:', user.id)
            # joint = user.skeleton.joints[Kinect.JOINT_TORSO]
            # x, y, z = joint.position.x, joint.position.y, joint.position.z
            x, y, z = user.centerOfMass.x, user.centerOfMass.y,\
                user.centerOfMass.z
            print('user pos:', (x, y, z))

        cv2.imshow('Video', np.hstack((rgb, dimg)))
        cv2.waitKey(60)

    kin.cleanup()
