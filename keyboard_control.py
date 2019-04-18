import curses
import numpy as np
import cv2
import slam
from robot import Robot

# Initialize curses.
stdscr = curses.initscr()

# Turns off automatic echoing of key press.
curses.noecho()

# React to keys instantly, without needing to press the return key.
curses.cbreak()

# To be passed to a wrapper function of curses. This avoid the terminal being
# messed up after application dies without restoring the terminal to its
# previous state.
def main(stdscr, robot):
    
    stdscr.nodelay(True)

    is_reverse = False
    is_key_press = False 
    v = 5
    w = 0.0001

    while True:
        try:
            stdscr.refresh()
            stdscr.addstr(0, 0, 'issued v, w: ' + str((v, w)))
            stdscr.clrtobot()
            stdscr.addstr(1, 0, 'read   v, w: ' + str(robot.get_motion()))
            stdscr.clrtobot()
            stdscr.addstr(2, 0, 'position   : ' + str(robot.get_pose()))
            stdscr.clrtobot()

            c = stdscr.getch()

            if c == ord('w'):
                # Set the robot to move forward.
                is_reverse = False
                is_key_press = True

            elif c == ord('s'):
                # Set the robot to move reverse.
                is_reverse = True
                is_key_press = True

            elif c == ord('a'):
                # Increase counterclockwise angular velocity.
                w += 0.1
                is_key_press = True

            elif c == ord('d'):
                # Increase clockwise angular velocity.
                w -= 0.1
                is_key_press = True

            elif c == ord('z'):
                # Decrease forward/reverse acceleration.
                v -= 0.5
                is_key_press = True

            elif c == ord('x'):
                # Increase forward/reverse acceleration.
                v += 0.5
                is_key_press = True

            elif c == ord(' '):
                # Abruptly stop the robot.
                v = 0
                w = 0
                is_key_press = True

            elif c == ord('q'):
                # Stop and quit.
                robot.halt()
                robot.stop_all_threads()
                break
            
            if is_key_press:
                if not is_reverse:
                    robot.drive(v, w)
                else:
                    robot.drive(-v, w)
                is_key_press = False

            # log_odds_vector = np.vectorize(slam.log_odds_to_prob)
            # m = log_odds_vector(robot.posterior)
            # m = m * 255
            # m = m.astype(np.uint8)
            # cv2.imshow('Posterior', m)

            # log_odds_vector = np.vectorize(slam.log_odds_to_prob)
            # d = log_odds_vector(robot.posterior)
            # d = d * 255
            # d = d.astype(np.uint8)

            # d = slam.d3_map(robot.posterior)
            # slam.draw_square(d, 10.0, robot.get_pose(), (255, 0, 0), width=3)
            # slam.draw_vertical_line(d, 250, (0, 0, 255))
            # slam.draw_horizontal_line(d, 250, (0, 0, 255))

            cv2.imshow('map', d)
            cv2.waitKey(250)

        except Exception as e:
            # No input.
            pass

# Initialize robot, automatically find and connect the serial port.
r = Robot(26) 
curses.wrapper(main, r)

# Terminate curses application.
curses.nocbreak()
curses.echo()
curses.endwin()

