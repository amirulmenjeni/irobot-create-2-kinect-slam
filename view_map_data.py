import cv2
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='View map images stored as numpy '\
        'file (*.npy).')
parser.add_argument('filename', help='Path to the numpy file.')
parser.add_argument('--write-video', action='store_true', default=False,\
        help='Write as .mp4 video.')
args = parser.parse_args()

def main(args):

    print('Numpy file:', args.filename)

    if args.write_video:
        frame_size = np.load(args.filename)[0].shape
        video_file = os.path.basename(args.filename)[:-4] + '.avi'
        video_writer = cv2.VideoWriter(\
            './videos/' + video_file,\
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,\
            (frame_size[1], frame_size[0])\
        )

    try:
        # This script cycles back to the first frame when the final frame is
        # shown. This flag check whether the full video is played at least once.
        is_completed_once = False
        while 1:
            frames = np.load(args.filename)
            for f in frames:
                cv2.imshow('image', f)
                if args.write_video and not is_completed_once:
                    video_writer.write(f)
                cv2.waitKey(100)
            is_completed_once = True
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main(args)
