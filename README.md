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

To install PyTorch from scratch, you will need a minimum of 5.5 GB of RAM (swap + real memory) with the settings below. If parts of the build fail, try reducing the number of jobs to 1 or 2. Ensure that you are also running a version of Python that will be compatible with the other packages - 3.8 is the latest recommended version. 3.9 and 3.10 may have compatibility issues with certain packages that we will use. The suggested method for handling multiple Python versions is to use `pyenv`. To install `pyenv`, use:

```
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install pyenv

# install stuff for the recommended build environment (debian)
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Then, add the following lines to your `~/.bashrc`:
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

If using SSH to log in, the `~/.bashrc` file may not be sourced. If you don't already have one, make a `~/.bash_profile` and make it source `~/.bashrc`:
```
touch ~/.bash_profile
echo "source ~/.bashrc" >> ~/.bash_profile
```

Now, restart your terminal (or source your updated `~/.bashrc`), and then install Python 3.8:
```
# 3.8.10 is fine too, 3.8.12 includes some security fixes
# install step can take quite a while
pyenv install 3.8.12
# go to where the HearingAId repo has been cloned
cd ~/HearingAId
# this sets the local version of Python for this folder to 3.8.12
pyenv local 3.8.12
# verify python version
python -V
# create new venv for the HearingAId repo
python -m venv venv
```


Finally, to build pytorch from source:

```
# Return to home directory
cd ~

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
export MAX_JOBS=2
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
# suggest using tab to autocomplete the file name
pip3 install ~/pytorch/dist/torch-1.10.0a0+git36449ea-cp38-cp38-linux_aarch64.whl

```
