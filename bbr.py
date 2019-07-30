import threading
import time
import queue

class Arbiter():

    arb_count = 0
    arbiters = []

    def __init__(self, robot):

        self.__id = Arbiter.arb_count
        self.__robot = robot
        self.current_behavior = None
        self.__timer = Timer()
        self.__is_locked = False
        self.next_queue = queue.Queue()

        Arbiter.arbiters.append(self)
        Arbiter.arb_count += 1
        
    def receive_request(self, behavior):

        if self.__is_locked:
            return

        """
        Whenever a behavior make a request, they're competing for the resources
        (i.e., the actuators whose action is controlled by te arbiter) in order
        to help the robot do its work.
        """

        if self.current_behavior is None:
            self.current_behavior = behavior

            try:
                # print 'STARTING THREAD:', self.current_behavior.get_name()
                self.current_behavior.start()
            except KeyboardInterrupt:
                pass

            return

        # Override the current behavior with the requesting behavior if the
        # requesting behavior has higher priority.
        if self.current_behavior.get_priority() < behavior.get_priority():

            # print '--> JOIN THREAD', self.current_behavior.get_name()
            self.current_behavior.stop()
            self.current_behavior.reset()
            self.current_behavior = None 
            self.current_behavior = behavior

            try:
                # print 'STARTING THREAD:', self.current_behavior.get_name()
                self.current_behavior.start()
            except KeyboardInterrupt:
                pass

        else:
            if not self.current_behavior.is_alive():
                # print 'BEHAVIOR NOT ALIVE:', self.current_behavior.get_name()
                self.current_behavior.reset()
                self.current_behavior = None

    def get_robot(self):
        return self.__robot

    def get_current_behavior_name(self):
        if self.current_behavior is not None:
            return self.current_behavior.get_name()
        else:
            return None

    def lock_others(self):
        self.current_behavior.stop()
        for arbiter in Arbiter.arbiters:
            if arbiter.__id == self.__id:
                continue
            arbiter.__is_locked = True

    def unlock_others(self):
        for arbiter in Arbiter.arbiters:
            arbiter.__is_locked = False

class Behavior():

    def __init__(self, name, arbiter, priority, func):

        """
        @func func: The function with parameters func(behavior, robot).
        """

        self.__name = name
        self.__func = None
        self.__init_func = None
        self.__arbiter = arbiter
        self.__priority = priority
        self.__interrupt = False
        self.__is_started = False
        self.__thread = None
        self.__input_param = None
        self.timer = Timer()

        self.set_function(func)

    def set_function(self, func, init_func=None):

        """
        Initialize the thread with the function that runs the behavior.
        """

        self.__func = func
        self.__init_func = init_func

        self.__thread = threading.Thread(
            target=self.__func, args=(self, self.__arbiter.get_robot()))

    def send_request(self):

        """
        Request the arbitrater to run this behavior any number of times.

        @param iteration: 
            The number of iterations that the arbiter run this behavior should
            the arbiter accept this behavior's request. If iteration is 0, it
            will run indefinitely until it is manually stopped by arbiter.stop()
            function. By default, only one iteration of the behavior is
            performed.
        """

        self.__arbiter.receive_request(self)

    def continue_to(self, behavior):

        """
        Once the current behavior ends, immediately start the next behavior set
        here.
        """

        self.__arbiter.next_queue.put(behavior)

    def get_name(self):

        """
        Get the name (identifier) of the behavior.
        """

        return self.__name

    def get_priority(self):

        """
        Get the priority of the behavior. Higher priority means more
        important.
        """

        return self.__priority

    def get_function(self):

        """
        Get the function that runs the behavior.
        """

        return self.__func

    def get_init_function(self):
        return self.__init_func

    def get_arbiter(self):

        """
        Get the arbiter of this behavior.
        """

        return self.__arbiter

    def get_param(self, param):

        """
        Get the argument parameters that goes into this behavior.
        """

        return self.__input_param[param]

    def interrupt(self):

        """
        Interrupt this behavior, stopping it. This is used to break an
        infinite loop within a function that runs the behavior (checked against
        the is_interrupted function).
        """

        self.__interrupt = True
        self.__is_stopped = True

    def is_interrupted(self):

        """
        A flag that checks if this behavior is to be interrupted. Used in a loop
        inside the function that runs the behavior.
        """

        if self.__interrupt:
            self.__interrupt = False # reset
            return True
        return False

    def is_started(self):

        if self.__is_started:
            self.__is_started = False
            return True
        return False

    def time_begin(self):
        return self.__time_begin

    def init(self):
        self.__init_func(self, self.__arbiter.get_robot())

    def start(self):
        self.__thread.start()

    def is_alive(self):
        return self.__thread.is_alive()

    def stop(self):
        self.interrupt()
        if self.is_alive():
            self.__thread.join()

    def reset(self):
        self.__thread = threading.Thread(
            target=self.__func, args=(self, self.__arbiter.get_robot()))

    def input_param(self, param):

        """
        Set the input parameters to this behavior, which may change the conduct
        of the running behavior.
        """

        self.__input_param = param

class Timer:

    def __init__(self):
        self.__begin = time.time()
        self.__duration = 0

    def timeup(self, duration):
        return time.time() - self.__begin > duration

    def reset(self):
        self.__begin = time.time()
