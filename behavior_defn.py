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
    COST_RADIUS = config.PATH_COST_RADIUS
    MAP_SIZE = config.GRID_MAP_SIZE
    RESOLUTION = config.GRID_MAP_RESOLUTION
    SPEED = config.NORMAL_DRIVE_SPEED
    ROTATE_SPEED = config.NORMAL_ROTATE_SPEED
    DELTA_TIME = config.CONTROL_DELTA_TIME

    CELL_STEPS = 3
    RADIUS_TOL = 150

    goal_cell = None
    next_cell = None
    goal_pos = None

    while True:

        robot_cell = robot.get_cell_pos()
        best_particle = robot.fast_slam.highest_particle()
        grid_map = np.copy(best_particle.m)

        if beh.is_interrupted():
            print('INTERRUPTED')
            break

        hum_pos = robot.h_t

        if len(hum_pos) == 0:
            print('No human.')
            break

        goal_cell = slam.world_to_cell_pos(hum_pos, MAP_SIZE, RESOLUTION)
        robot.goal_cell = goal_cell
        robot.nearest_human = goal_cell

        # We're quite sure that since the robot is at position (ru, rv),
        # there's no obstacles around here.
        grid_map = slam.fill_area(grid_map, robot_cell, KERNEL_RADIUS + 1, 0.10)

        solution = slam.shortest_path(robot_cell, goal_cell, grid_map,
            OCCU_THRES, kernel_radius=KERNEL_RADIUS, cost_radius=COST_RADIUS,
            epsilon=2)
        robot.current_solution = solution

        try:
            if CELL_STEPS - 1 < len(solution):
                next_cell = solution[CELL_STEPS - 1]
            else:
                next_cell = solution[-1]
        except IndexError:

            print('APPROACH-HUMAN: Error getting next cell.')

            slam.fill_area(robot.hum_grid_map, nearest_human, 5, 0,\
                out=robot.hum_grid_map)

            next_cell = None
            robot.nearest_human = None
            continue

        goal_pos = slam.cell_to_world_pos(goal_cell, MAP_SIZE, RESOLUTION)

        if rutil.is_in_circle(goal_pos, RADIUS_TOL, best_particle.x[:2]):
            print('Human approached.')
            robot.drive_velocity(0, 0)
            robot.goal_cell = None
            robot.nearest_human = None
            flag_approached = True
            playsound(config.SND_GREET)
            break
        else:
            next_pos = slam.cell_to_world_pos(next_cell, MAP_SIZE, RESOLUTION)
            turn_radius = robot.inverse_drive_kinematic(next_pos)
            robot.drive_radius(SPEED, turn_radius)

        time.sleep(DELTA_TIME)

    print('<< APPROACH-HUMAN')

def explore(beh, robot):

    playsound(config.SND_EXPLORE)

    print('>> EXPLORE')

    OCCU_THRES = config.OCCU_THRES
    KERNEL_RADIUS = config.BODY_KERNEL_RADIUS
    COST_RADIUS = config.PATH_COST_RADIUS
    MAP_SIZE = config.GRID_MAP_SIZE
    RESOLUTION = config.GRID_MAP_RESOLUTION
    SPEED = config.NORMAL_DRIVE_SPEED
    DELTA_TIME = config.CONTROL_DELTA_TIME

    CELL_STEPS = 3
    RADIUS_TOL = 15

    # The first unexplored cell found in the solution path.
    frontier_cell = None

    goal_cell = None
    next_cell = None
    solution = []

    while True:

        if beh.is_interrupted():
            print('INTERRUPTED')
            break

        robot_cell = robot.get_cell_pos()
        best_particle = robot.fast_slam.highest_particle()
        grid_map = np.copy(best_particle.m)
        entr_map = slam.entropy_map(grid_map)

        # We're quite sure that since the robot is at position (ru, rv),
        # there's no obstacles around here. This also prevent the robot getting
        # stuck due to A* search condition when an obstacle "spawns" very close
        # to the robot.
        # rv, ru = robot_cell
        # for i in range(rv - KERNEL_RADIUS, rv + KERNEL_RADIUS + 1):
        #     for j in range(ru - KERNEL_RADIUS, ru + KERNEL_RADIUS + 1):
        #         grid_map[i, j] = 0.1

        if (next_cell is None) or (goal_cell is None):

            robot.drive_velocity(0, 0)
            goal_cell, solution = robot.plan_explore(KERNEL_RADIUS, COST_RADIUS)

            # Get the cell on the solution that is an unexplored cell.
            frontier_cell = slam.exploration_path_frontier_cell(
                grid_map, solution)

            robot.goal_cell = frontier_cell
            robot.current_solution = solution
        else:
            # Update solution when next_cell is reached.
            if slam.goal_test(robot_cell, next_cell, KERNEL_RADIUS):
                solution = slam.shortest_path(robot_cell, goal_cell,\
                    grid_map, OCCU_THRES, kernel_radius=KERNEL_RADIUS,\
                    cost_radius=COST_RADIUS, epsilon=2)
                frontier_cell = slam.exploration_path_frontier_cell(
                    grid_map, solution)
                robot.goal_cell = frontier_cell
                robot.current_solution = solution
        
        try:
            if CELL_STEPS - 1 < len(solution):
                next_cell = solution[CELL_STEPS - 1]
            else:
                next_cell = solution[-1]
        except IndexError:
            print('EXPLORE: Error getting next cell.')
            next_cell = None
            continue

        goal_pos = slam.cell_to_world_pos(goal_cell, MAP_SIZE, RESOLUTION)

        # If the robot is within 1 m distance to a frontier, do a 360 scan.
        if frontier_cell is not None:
            dist = rutil.euclidean_distance(frontier_cell, robot_cell)
            if RESOLUTION * dist < 100:
                beh.continue_to(robot.behaviors[Beh.SCAN_360])
                robot.current_solution = []
                break

        # If the robot have traveled more than 250 cm, do a 360 scan.
        if robot.get_distance_traveled() > 250:
            print('Scanning after 250 cm distance traveled.')
            robot.reset_distance_traveled()
            beh.continue_to(robot.behaviors[Beh.SCAN_360])
            return

        if rutil.is_in_circle(goal_pos, RADIUS_TOL, best_particle.x[:2]):
            print('Reached goal (within circle radius).')
            robot.goal_cell = None
            return
        else:
            next_pos = slam.cell_to_world_pos(next_cell, MAP_SIZE, RESOLUTION)
            turn_radius = robot.inverse_drive_kinematic(next_pos)
            robot.drive_radius(SPEED, turn_radius)

        time.sleep(DELTA_TIME)

    print('<< EXPLORE')

def scan_360(beh, robot):

    print('>> SCAN-360')

    ROTATE_SPEED = config.NORMAL_ROTATE_SPEED
    DELTA_TIME = config.CONTROL_DELTA_TIME
    
    robot.drive_velocity(0, 0)
    time.sleep(0.5)
    robot.drive_velocity(0, ROTATE_SPEED)

    total_angle = 0

    timer = Timer()

    while True:

        robot.wait_motion_update()

        if beh.is_interrupted():
            robot.drive_velocity(0, 0)
            break

        if total_angle > 2*math.pi:
            robot.drive_velocity(0, 0)
            break

        total_angle += abs(robot.get_delta_angle())

    print('<< SCAN-360')

def go_to_input_goal(beh, robot):

    print('>> GO-TO-INPUT-GOAL')

    OCCU_THRES = config.OCCU_THRES
    KERNEL_RADIUS = config.BODY_KERNEL_RADIUS
    COST_RADIUS = config.PATH_COST_RADIUS
    MAP_SIZE = config.GRID_MAP_SIZE
    RESOLUTION = config.GRID_MAP_RESOLUTION
    SPEED = config.NORMAL_DRIVE_SPEED
    DELTA_TIME = config.CONTROL_DELTA_TIME

    CELL_STEPS = 3
    RADIUS_TOL = 15

    goal_cell = beh.get_param('goal-cell')
    robot.goal_cell = goal_cell
    next_cell = None
    solution = []

    while True:

        robot.wait_motion_update()

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
        # for i in range(rv - KERNEL_RADIUS, rv + BODY_RADIUS_CELL + 1):
        #     for j in range(ru - BODY_RADIUS_CELL, ru + BODY_RADIUS_CELL + 1):
        #         grid_map[i, j] = 0.05
        grid_map[rv, ru] = 0.05

        if next_cell is None:
            solution = slam.shortest_path(robot_cell, goal_cell,
                grid_map, OCCU_THRES, kernel_radius=KERNEL_RADIUS,\
                cost_radius=COST_RADIUS)
            robot.current_solution = solution
        else:
            if slam.goal_test(robot_cell, next_cell, KERNEL_RADIUS):
                solution = slam.shortest_path(robot_cell, goal_cell,\
                    grid_map, OCCU_THRES, kernel_radius=KERNEL_RADIUS,\
                    cost_radius=COST_RADIUS)
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

    print('<< GO-TO-INPUT-GOAL')

def manual_driving(beh, robot):

    print('>> MANUAL-DRIVING')

    SPEED_V = config.NORMAL_DRIVE_SPEED
    SPEED_W = config.NORMAL_ROTATE_SPEED

    input_v = beh.get_param('v')
    input_w = beh.get_param('w')

    robot.drive_velocity(input_v, input_w)

    print('<< MANUAL-DRIVING')

def stop_driving(beh, robot):

    print('>> STOP-DRIVING')

    robot.current_solution = []
    robot.goal_cell = None
    robot.drive_velocity(0, 0)
    
    print('<< STOP-DRIVING')

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

        if timer.timeup(3):
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
    MANUAL_DRIVING = auto()
    STOP_DRIVING = auto()
    SCAN_360 = auto()

beh_def_list = [\
    (Beh.EXPLORE, 0, explore),
    (Beh.APPROACH_HUMAN, 200, approach_human2),
    (Beh.SCAN_360, 300, scan_360),
    (Beh.ESCAPE_OBSTACLE, 999, escape_obstacle),
    (Beh.GO_TO_INPUT_GOAL, 2000, go_to_input_goal),
    (Beh.MANUAL_DRIVING, 2001, manual_driving),
    (Beh.STOP_DRIVING, 3001, stop_driving),
]
