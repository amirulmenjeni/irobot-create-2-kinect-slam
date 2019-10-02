#Installation

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
