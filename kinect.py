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

    MAX_RANGE = 2**12 - 1

    def __init__(self, redist, video_shape=(480, 640), depth_shape=(480, 640)):

        self.video_h = video_shape[0]
        self.video_w = video_shape[1]
        self.depth_h = depth_shape[0]
        self.depth_w = depth_shape[1]

        openni2.initialize(redist)
        nite2.initialize(redist)
        
        self.dev = openni2.Device.open_any()

        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream = self.dev.create_color_stream()

        self.depth_stream.set_mirroring_enabled(False)
        self.color_stream.set_mirroring_enabled(False)

        self.depth_stream.set_video_mode(\
            _openni2.OniVideoMode(\
            pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,\
            resolutionX=self.depth_w,
            resolutionY=self.depth_h,
            fps=30))

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
        self.dev.set_depth_color_sync_enabled(False)

        # Align depth to rgb.
        self.dev.set_image_registration_mode(\
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    def get_rgb(self):

        data = self.color_stream.read_frame().get_buffer_as_uint8()
        bgr = np.fromstring(data, dtype=np.uint8).reshape(480, 640, 3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def get_depth(self):

        data = self.depth_stream.read_frame().get_buffer_as_uint16()
        dmap = np.fromstring(data, dtype=np.uint16).reshape(480, 640)

        # No reading set to max depth.
        dmap[abs(dmap - 0) < 1e-6] = Kinect.MAX_RANGE

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

    def get_users_pos(self):

        users = self.get_users()
        positions = []

        for i in range(len(users)):

            cmass = users[i].centerOfMass
            x, y, z = cmass.x, cmass.y, cmass.z

            # Ignore erroneous position data where x ~= 0 and z ~= 0.
            if abs(x - 0) >= 1e-6 and abs(z - 0) >= 1e-6:
                positions.append([z, -x])

        positions = np.array(positions)

        return positions

    def get_depth_fps(self):

        return self.depth_stream.get_video_mode().fps

    def get_color_fps(self):

        return self.color_stream.get_video_mode().fps

    def depth_display(depth_map):

        d = np.uint8(depth_map.astype(float) * 255 / Kinect.MAX_RANGE)
        d = 255 - cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)
        return d

    def depth_map_to_world(self, depth_map, slice_row=-1):

        if slice_row < 0:
            depth_slice = np.min(depth_map, axis=0)[np.newaxis]
        else:
            depth_slice = depth_map[slice_row, :][np.newaxis]

        world_xyz = [0] * self.depth_w

        # Set erroneous value (max range) to 0.
        depth_slice[depth_slice == Kinect.MAX_RANGE] = 0

        for u in range(self.depth_w):
            world_xyz[u] = openni2.convert_depth_to_world(\
                self.depth_stream, u, 0, depth_slice[0][u])

        world_xyz = np.array(world_xyz)

        x, y, z = world_xyz.T
        obstacles_xy = np.hstack((z.T[:, np.newaxis], -x.T[:, np.newaxis]))
        obstacles_xy *= 0.1 # mm to cm

        return obstacles_xy

    def clean_up(self):

        self.color_stream.stop()
        self.depth_stream.stop()
        openni2.unload()
        nite2.unload()

if __name__ == '__main__':

    kin = Kinect('./redist')

    try:
        while 1:
            rgb = kin.get_rgb()

            dmap = kin.get_depth()
            dimg = Kinect.depth_display(dmap)

            for upos in kin.get_users_pos():
                print('wpos:', upos)

            print('depth:', kin.depth_stream.get_video_mode().fps)
            print('color:', kin.color_stream.get_video_mode().fps)

            cv2.imshow('Video', np.hstack((rgb, dimg)))
            cv2.waitKey(60)

    except KeyboardInterrupt:
        kin.cleanup()
