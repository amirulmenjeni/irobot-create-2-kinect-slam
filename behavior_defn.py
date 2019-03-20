"""
Define behavior functions here. Each behavior function should define actions per
iteration.
"""

import time
import rutil
import config
import slam
import math
import numpy as np
from bbr import Timer
from playsound import playsound
from enum import Enum, auto

def approach_human(beh, robot):

    playsound(config.SND_APPROACH)

    print('>> APPROACH-HUMAN')

    OCCU_THRES = config.OCCU_THRES
    KERNEL_RADIUS = config.BODY_KERNEL_RADIUS
    MAP_SIZE = config.GRID_MAP_SIZE
    RESOLUTION = config.GRID_MAP_RESOLUTION
    SPEED = config.NORMAL_DRIVE_SPEED
    DELTA_TIME = config.CONTROL_DELTA_TIME

    CELL_STEPS = 3
    RADIUS_TOL = 50

    goal_cell = None
    next_cell = None
    goal_pos = None
    flag_approached = False

    while True:

        robot_cell = robot.get_cell_pos()
        best_particle = robot.fast_slam.highest_particle()
        grid_map = best_particle.m

        if beh.is_interrupted():
            print('INTERRUPTED')
            break

        nearest_human = robot.get_nearest_human()
        if nearest_human is None:
            print('NO MORE VISIBLE HUMAN.')
            break
        robot.nearest_human = nearest_human

        # We're quite sure that since the robot is at position (ru, rv),
        # there's no obstacles around here.
        grid_map = slam.fill_area(grid_map, robot_cell, KERNEL_RADIUS + 1, 0.10)

        goal_cell = nearest_human
        robot.goal_cell
        solution = slam.shortest_path(robot_cell, goal_cell, grid_map,
            OCCU_THRES, kernel_radius=KERNEL_RADIUS)
        robot.current_solution = solution

        try:
            if CELL_STEPS - 1 < len(solution):
                next_cell = solution[CELL_STEPS - 1]
            else:
                next_cell = solution[-1]
        except IndexError:

            # print('APPROACH-HUMAN: Error getting next cell.')

            slam.fill_area(robot.hum_grid_map, nearest_human, 5, 0,\
                out=robot.hum_grid_map)

            next_cell = None
            continue

        goal_pos = slam.cell_to_world_pos(goal_cell, MAP_SIZE, RESOLUTION)

        if rutil.is_in_circle(goal_pos, RADIUS_TOL, best_particle.x[:2]):
            playsound(config.SND_GREET)
            robot.goal_cell = None
            flag_approached = True
            break
        else:
            next_pos = slam.cell_to_world_pos(next_cell, MAP_SIZE, RESOLUTION)
            turn_radius = robot.inverse_drive_kinematic(next_pos)
            robot.drive_radius(SPEED, turn_radius)

        time.sleep(DELTA_TIME)

    if flag_approached:

        # Orient the robot to face the human once it has approached the human.
        while True:

            best_particle = robot.fast_slam.highest_particle()
            pose = best_particle.x

            dir_vec = rutil.direction_vector(pose[:2], goal_pos)
            hdg_vec = rutil.angle_to_dir(pose[2])

            heading_diff = rutil.angle_error(hdg_vec, dir_vec)

            w = 0.0873 * math.sin(abs(heading_diff) / 2.0)

            if heading_diff > 0:
                robot.drive_velocity(0, +w)
            elif heading_diff < 0:
                robot.drive_velocity(0, -w)

            if abs(heading_diff) < 0.043633:
                break

            time.sleep(DELTA_TIME)

    print('<< APPROACH-HUMAN')

def explore(beh, robot):

    playsound(config.SND_EXPLORE)

    print('>> EXPLORE')

    OCCU_THRES = config.OCCU_THRES
    KERNEL_RADIUS = config.BODY_KERNEL_RADIUS
    MAP_SIZE = config.GRID_MAP_SIZE
    RESOLUTION = config.GRID_MAP_RESOLUTION
    SPEED = config.NORMAL_DRIVE_SPEED
    DELTA_TIME = config.CONTROL_DELTA_TIME

    CELL_STEPS = 3
    RADIUS_TOL = 15

    goal_cell = None
    next_cell = None
    solution = []

    while True:

        if beh.is_interrupted():
            print('INTERRUPTED')
            break

        robot_cell = robot.get_cell_pos()
        best_particle = robot.fast_slam.highest_particle()
        grid_map = best_particle.m
        entr_map = slam.entropy_map(grid_map)

        # We're quite sure that since the robot is at position (ru, rv),
        # there's no obstacles around here. This also prevent the robot getting
        # stuck due to A* search condition when an obstacle "spawns" very close
        # to the robot.
        rv, ru = robot_cell
        for i in range(rv - KERNEL_RADIUS, rv + KERNEL_RADIUS + 1):
            for j in range(ru - KERNEL_RADIUS, ru + KERNEL_RADIUS + 1):
                grid_map[i, j] = 0.1

        if (next_cell is None) or (goal_cell is None):
            goal_cell, solution = robot.plan_explore(KERNEL_RADIUS)
            robot.goal_cell = goal_cell
            robot.current_solution = solution
        else:
            if slam.goal_test(robot_cell, next_cell, KERNEL_RADIUS):
                solution = slam.shortest_path(robot_cell, goal_cell,\
                    grid_map, OCCU_THRES, kernel_radius=KERNEL_RADIUS)
                robot.current_solution = solution

        goal_pos = slam.cell_to_world_pos(goal_cell, MAP_SIZE, RESOLUTION)

        try:
            if CELL_STEPS - 1 < len(solution):
                next_cell = solution[CELL_STEPS - 1]
            else:
                next_cell = solution[-1]
        except IndexError:
            print('EXPLORE: Error getting next cell.')
            next_cell = None
            continue

        if rutil.is_in_circle(goal_pos, RADIUS_TOL, best_particle.x[:2]):
            robot.test_song()
            robot.goal_cell = None
            return
        else:
            next_pos = slam.cell_to_world_pos(next_cell, MAP_SIZE, RESOLUTION)
            turn_radius = robot.inverse_drive_kinematic(next_pos)
            robot.drive_radius(SPEED, turn_radius)

        time.sleep(DELTA_TIME)

    print('<< EXPLORE')

def go_to_input_goal(beh, robot):

    print('>> GO-TO-INPUT-GOAL')

    OCCU_THRES = config.OCCU_THRES
    KERNEL_RADIUS = config.BODY_KERNEL_RADIUS
    MAP_SIZE = config.GRID_MAP_SIZE
    RESOLUTION = config.GRID_MAP_RESOLUTION
    SPEED = config.NORMAL_DRIVE_SPEED
    DELTA_TIME = config.CONTROL_DELTA_TIME

    CELL_STEPS = 3
    RADIUS_TOL = 30

    goal_cell = beh.get_param('goal-cell')
    robot.goal_cell = goal_cell
    next_cell = None
    solution = []

    while True:

        if beh.is_interrupted():
            print('INTERRUPTED')
            break

        robot_cell = robot.get_cell_pos()
        best_particle = robot.fast_slam.highest_particle()
        grid_map = best_particle.m

        # We're quite sure that since the robot is at position (ru, rv),
        # there's no obstacles around here. This also prevent the robot getting
        # stuck due to A* search condition when an obstacle "spawns" very close
        # to the robot.
        rv, ru = robot_cell
        for i in range(rv - KERNEL_RADIUS, rv + KERNEL_RADIUS + 1):
            for j in range(ru - KERNEL_RADIUS, ru + KERNEL_RADIUS + 1):
                grid_map[i, j] = 0.05

        if next_cell is None:
            solution = slam.shortest_path(robot_cell, goal_cell,
                grid_map, OCCU_THRES, kernel_radius=KERNEL_RADIUS)
            robot.current_solution = solution
        else:
            if slam.goal_test(robot_cell, next_cell, KERNEL_RADIUS):
                solution = slam.shortest_path(robot_cell, goal_cell,\
                    grid_map, OCCU_THRES, kernel_radius=KERNEL_RADIUS)
                robot.current_solution = solution

        goal_pos = slam.cell_to_world_pos(goal_cell, MAP_SIZE, RESOLUTION)

        try:
            if CELL_STEPS - 1 < len(solution):
                next_cell = solution[CELL_STEPS - 1]
            else:
                next_cell = solution[-1]
        except IndexError:
            print('GO-TO-INPUT-GOAL: Error getting next cell.')
            next_cell = None
            robot.drive_velocity(0, 0)
            break

        if rutil.is_in_circle(goal_pos, RADIUS_TOL, best_particle.x[:2]):
            robot.test_song()
            robot.drive_velocity(0, 0)
            robot.goal_cell = None
            print('Reached goal destination.')
            break
        else:
            next_pos = slam.cell_to_world_pos(next_cell, MAP_SIZE, RESOLUTION)
            turn_radius = robot.inverse_drive_kinematic(next_pos)
            robot.drive_radius(SPEED, turn_radius)

        time.sleep(DELTA_TIME)

    print('<< GO-TO-INPUT-GOAL')

def escape_obstacle(beh, robot):

    playsound(config.SND_OOPS)

    print('>> ESCAPE-OBSTACLE')

    SPEED = config.ESCAPE_OBSTACLE_SPEED
    DELTA_TIME = config.CONTROL_DELTA_TIME

    robot_cell = robot.get_cell_pos()
    prev_cell = robot_cell

    timer = Timer()

    while True:

        if beh.is_interrupted():
            break

        if timer.timeup(2):
            robot.drive_velocity(0, 0)
            break

        robot.drive_velocity(-SPEED, 0)

        time.sleep(DELTA_TIME)

    print('<< ESCAPE-OBSTACLE')

##################################################
# Define the name, priority, and the function of each behavior. This will be
# used for initializing the robot behavior.
##################################################

class Beh(Enum):
    EXPLORE = auto()
    APPROACH_HUMAN = auto()
    ESCAPE_OBSTACLE = auto()
    GO_TO_INPUT_GOAL = auto()

beh_def_list = [\
    (Beh.EXPLORE, 0, explore),
    # (Beh.APPROACH_HUMAN, 200, approach_human),
    (Beh.ESCAPE_OBSTACLE, 999, escape_obstacle),
    (Beh.GO_TO_INPUT_GOAL, 1000, go_to_input_goal)
]
