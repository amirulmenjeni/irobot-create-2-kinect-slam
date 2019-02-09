import numpy as np
import cv2
import slam

def main():

    """
    Demo of probabilistic motion model.
    """

    curr_pose = np.array([0, 0, 0])
    control = np.array([550, 0])
    dt = 0.25

    grid_map = np.zeros((500, 500), dtype=np.uint32)
    grid_map.fill(255)

    # Forward velocity noise.
    a1 = 1e-3
    a2 = 1e-3

    # Rotational velocity noise.
    a3 = 3e-6
    a4 = 3e-6

    # Perturbation noise.
    a5 = 3e-4
    a6 = 3e-4
    noise = (a1, a2, a3, a4, a5, a6)

    sz = 5000
    samples = np.array([[0, 0]] * sz)
    for i in range(sz):
        samples[i] = slam.sample_motion_model_velocity(control, curr_pose,
                dt, noise=noise)[:2]

    print(samples)

    sample_cells = slam.world_frame_to_cell_pos(samples,\
        grid_map.shape, 1)

    # Count samples.
    sample_count = {}

    d = np.dstack((grid_map, grid_map, grid_map))

    min_count = +np.inf
    max_count = -np.inf

    for row, col in sample_cells:

        p = (row, col)

        if p not in sample_count:
            sample_count[p] = 1
        else:
            sample_count[p] += 1

        if sample_count[p] < min_count:
            min_count = sample_count[p]
        elif sample_count[p] > max_count:
            max_count = sample_count[p]

    print('min_count:', min_count)
    print('max_count:', max_count)

    for row, col in sample_cells:

        p = (row, col)

        gray_level = 255 * (sample_count[p] - min_count) /\
                (max_count - min_count)
        gray_level = int(gray_level)
        d[p] = (gray_level, gray_level, gray_level)

    slam.draw_square(d, 10.0, curr_pose, (255, 0, 0), width=1)

    cv2.imshow('test', d.astype(np.uint8))

    while 1:
        if cv2.waitKey(10) == ord('q'):
            break

if __name__ == '__main__':
    main()
