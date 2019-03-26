import sys
import numpy as np
import cv2
import slam
from openni import openni2, nite2, _openni2, _nite2, utils

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
    MIN_DEPTH_MM = 600
    MAX_DEPTH_MM = 8000

    def __init__(self, redist, video_shape=(480, 640), depth_shape=(480, 640),
        enable_color_stream=False):

        self.color_stream_enabled = enable_color_stream

        self.video_h = video_shape[0]
        self.video_w = video_shape[1]
        self.depth_h = depth_shape[0]
        self.depth_w = depth_shape[1]

        openni2.initialize(redist)
        nite2.initialize(redist)
        
        self.dev = openni2.Device.open_any()

        if enable_color_stream:
            self.color_stream = self.dev.create_color_stream()
            self.color_stream.set_mirroring_enabled(False)
            self.color_stream.start()

        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.set_mirroring_enabled(False)
        self.depth_stream.set_video_mode(\
            _openni2.OniVideoMode(\
            pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,\
            resolutionX=self.depth_w,\
            resolutionY=self.depth_h,\
            fps=30))
        self.depth_stream.start()

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
        # self.dev.set_image_registration_mode(\
        #     openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    def get_rgb(self):

        if self.color_stream_enabled:
            data = self.color_stream.read_frame().get_buffer_as_uint8()
            bgr = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        else:
            return None

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

                if user.state == nite2.UserState.NITE_USER_STATE_VISIBLE:
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

        positions = np.array(positions) * 0.1

        return positions

    def get_depth_fps(self):

        return self.depth_stream.get_video_mode().fps

    def get_color_fps(self):

        if self.color_stream_enabled:
            return self.color_stream.get_video_mode().fps
        return None

    def depth_display(depth_map, clean=False):

        img = np.copy(depth_map.astype(np.float32))

        if not clean:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)

            if min_val < max_val:
                img = (img - 0) / (10000 - 0)

        else:
            x_ij, y_ij = Kinect.xy_map(depth_map)

            img = Kinect.cleaned_depth_map(depth_map, y_ij)

            RANGE = Kinect.MAX_DEPTH_MM - Kinect.MIN_DEPTH_MM
            img = (img - Kinect.MIN_DEPTH_MM) / RANGE

            img = img.astype(np.float32)

        d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return d

    def cleaned_depth_map(depth_map, Y):

        """
        Attempt to clean the depth_map from noise. Y is 480x640 array that
        maps each pixel to the real-world y-coordinates.
        """

        depth_map = np.copy(depth_map)

        # Filter out values outside accepted range.
        depth_map[depth_map <= Kinect.MIN_DEPTH_MM] = 10000
        depth_map[depth_map >= Kinect.MAX_DEPTH_MM] = 10000

        # Filter out depth values above the robot's height.
        v, u = np.where(Y >= 70.0)
        depth_map[v, u] = 10000

        # Filter out depth values below certain height.
        v, u = np.where(Y <= -75.0)
        depth_map[v, u] = 10000

        return depth_map

    def depth_map_to_world(self, depth_map, clean=False, slice_row=240):

        if clean:
            _, Y = Kinect.xy_map(depth_map)
            depth_map = Kinect.cleaned_depth_map(depth_map, Y)
            depth_slice = np.min(depth_map, axis=0)[np.newaxis]
        else:
            depth_slice = depth_map[slice_row, :][np.newaxis]

        world_xyz = []

        for u in range(self.depth_w):
            if Kinect.MIN_DEPTH_MM < depth_slice[0][u] < Kinect.MAX_DEPTH_MM:
                xyz = openni2.convert_depth_to_world(\
                    self.depth_stream, u, 0, depth_slice[0][u])
                world_xyz.append(xyz)

        world_xyz = np.array(world_xyz)

        obstacles_xy = np.full((0, 2), 0)

        if len(world_xyz) > 0:
            x, y, z = world_xyz.T
            obstacles_xy = np.hstack((z.T[:, np.newaxis], -x.T[:, np.newaxis]))
            obstacles_xy *= 0.1 # mm to cm

        return obstacles_xy

    def xy_map(depth_map):

        """
        Returns X and Y, each of which is a 480x640 array which maps each pixel
        to their respective world-coordinates.
        """

        w = depth_map.shape[1]
        h = depth_map.shape[0]

        i, j = np.mgrid[:h, :w]

        # Constants multiplier to get world x- and y-coordinates.
        X_MULT = 1.12032
        Y_MULT = 0.84824

        j_norm = (j - 0) / w
        i_norm = (i - 0) / h

        X = (j_norm - 0.5) * (320 / w) * X_MULT * depth_map
        Y = (i_norm - 0.5) * (240 / h) * Y_MULT * depth_map

        return X, Y

    def clean_up(self):

        if self.color_stream_enabled:
            self.color_stream.stop()
        self.depth_stream.stop()
        openni2.unload()
        nite2.unload()

if __name__ == '__main__':

    kin = Kinect('./redist', enable_color_stream=True)

    try:
        while 1:
            rgb = kin.get_rgb()

            dmap = kin.get_depth()
            dimg = Kinect.depth_display(dmap)

            cv2.imshow('Color', rgb)
            cv2.imshow('Depth', dimg)
            cv2.waitKey(150)

    except KeyboardInterrupt:
        kin.clean_up()
