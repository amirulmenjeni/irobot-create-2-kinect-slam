# Installation

1. Clone the git repository:
   `git clone https://gitlab.com/ailabuser/kinect-roomba-slam`
2. Install `libudev`:
    - On Ubuntu: `sudo apt install libudev-dev`
    - On Arch Linux: `sudo pacman -S systemd-libs`
    - Create a symbolic link to `libudev.so` in the `redist` directory.
3. Install `python3-pip`
4. Install `virtualenv`
5. Create a new virtual `python3` environment inside the directory of the cloned
   repository:

   `virtualenv -p /usr/bin/python3 venv`

   This will create a new directory called `venv` within the repository.
6. Activate the new virtual environment:

   `source venv/bin/activate`
7. Install dependencies (while still having the virtual environment activated):

   `pip3 install -r requirements.txt`

# Running the robot

To run the SLAM robot, the a laptop mounted on the robot must run the core
script which acts as the server; a remote client can then connect to this
server to remotely view and interact with the map.

1. Run `start.py` script to start the SLAM system on the laptop mounted on the
   robot (pass `--disable-auto`, because by default it will try to explore 
   autonomously, which is currently not up to mark):

```bash
python start.py --disable-auto
```

2. Run `client.py` script to view the map (and the robot location) on another
   laptop or PC. You can also interact with the map to manually pick a point
   location on the map for the robot to autonomously drive to. You'll also need
   to pass the IP address of the laptop running the core script:

```bash
python client.py -a IPV4_ADDRESS
```
