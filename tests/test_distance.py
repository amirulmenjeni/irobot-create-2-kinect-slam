from robot import Robot
from time import sleep

# Test the forward velocity reading of the robot. In this case,
# set the forward speed to 5 cm/s, and set the delay for 20 seconds.
# After the 20 seconds delay, the robot should stop at about 100 cm mark.

r = Robot()
r.drive(5, 0)
sleep(20)
r.halt()
r.stop_all_threads()
