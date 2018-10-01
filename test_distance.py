from robot import Robot
from time import sleep

r = Robot()
r.drive(5, 0)
sleep(20)
r.halt()
r.stop_all_threads()
