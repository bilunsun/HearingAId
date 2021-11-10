# HearingAId

If UserWarning: No audio backend is available. then `pip install PySoundFile`


## Windows 10 Setup Troubleshooting
Some things encountered along the way.
### Installing PyAudio
To install PyAudio, run:
```
pip install pipwin
pipwin install pyaudio
```

## Raspberry Pi 4 Setup
Primary challenge encountered is the installation of Pytorch.
To install Pytorch, I built it from the pytorch repo, because the architecture of the Pi 4 is `aarch64`, and there are no available wheels for this architecture from pip.

To install PyTorch from scratch, you will need a minimum of 5.5 GB of RAM (swap + real memory) with the settings below. If parts of the build fail, try reducing the number of jobs to 1 or 2. Ensure that you are also running a version of Python that will be compatible with the other packages - 3.8 is the latest recommended version. 3.9 and 3.10 may have compatibility issues with certain packages that we will use.

```
# The usual sudo update and upgrade
sudo apt-get update && sudo apt-get upgrade

# Clone the 1.10 branch of the pytorch repository (can take a while)
git clone -b v1.10.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git

# Go into the directory, make venv and activate
cd pytorch
python3 -m venv venv
source venv/bin/activate

# Install dependencies
sudo apt-get install python3-dev
sudo apt-get install ninja-build git cmake
sudo apt-get install libopenmpi-dev libomp-dev ccache
sudo apt-get install libopenblas-dev libblas-dev libeigen3-dev

# Update python dependencies
pip3 install -U wheel mock pillow
pip3 install -U setuptools

# Install requirements.txt
pip3 install -r requirements.txt

# Temporary environment variables for build
export BUILD_CAFFE2_OPS=OFF
export USE_FBGEMM=OFF
export USE_FAKELOWP=OFF
export BUILD_TEST=OFF
export USE_MKLDNN=OFF
export USE_NNPACK=ON
export USE_XNNPACK=ON
export USE_QNNPACK=ON
export MAX_JOBS=4
export USE_OPENCV=OFF
export USE_NCCL=OFF
export USE_SYSTEM_NCCL=OFF
PATH=/usr/lib/ccache:$PATH

# Clean up previous build (if necessary)
python3 setup.py clean

# Start build
python3 setup.py bdist_wheel

# Install wheel (enable virtual env that will use it first)
cd ~/HearingAId
source venv/bin/activate
pip3 install ~/pytorch/dist/torch-1.10.0a0+git36449ea-cp39-cp39-linux_aarch64.whl

```
