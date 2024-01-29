# Code to reproduce experiments from the paper
## Quick setup
To clone this repo to your local machine, type this command from your preferred directory:
```
git clone https://github.com/avysogorets/preferans-solver.git
```
Then, follow these steps in your terminal window to set up virtual environment:
#### MacOS & Linux
```
python -m pip install --user --upgrade pip # install pip
python -m pip install --user virtualenv # install environment manager
python -m venv env # create a new environment
source env/bin/activate # activate the environment
python -m pip install -r requirements.txt # install packages
```
## Usage
The supported models include LeNet-300-100 (FashionMNIST) and LeNet-5 (CIFAR-10). For demonstration
purposes, we recommend using model ```Demo``` and a small 2D dataset ```Circles```. See the Jupyter notebook for
most of the experiments. To replicate the empirical work from Section 5, please see ```trinability.py``` and
associated command line arguments.
