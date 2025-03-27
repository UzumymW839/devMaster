# 2023_BinauralSpeechEnhancement
Author: Mattes Ohlenbusch (ohlems@idmt.fhg.de)


## Description
- Deep speech enhancement for binaural hearables
- Multiple input channels, one output channel (for now)
- Speech and noise are recorded simultaneously, speech is assumed to arrive from the front, noise can be diffuse or directional, coming from any direction in the horizontal plane
- Data generation of multi-channel speech+noise files
- Training and testing of DNN-based enhancement models


## Usage
- Generate training and testing data using the scripts in `scripts/`
- Training using `train.py -c default_config.json` or a different config file
- Testing a trained model with `test.py -c saved/models/0717_100954/config.json -t test_configs/snr_test.json` where the argument `-c` the config file of a specific run is given as argument and the argument `-t` corresponds to the config file for the test to run on the trained model
- Monitor ongoing runs with tensorboard by running `tensorboard --logdir=saved/`, access for example via ssh+port forwarding: `ssh -L 6006:localhost:6006 olserv05`


## Project status
Currently, this repository is set up to investigate feasibility of training a "small" model to enable real-time speech enhancement with binaural hearable inputs.
