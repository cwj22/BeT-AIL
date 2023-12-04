# BeT-AIL


Code for "BeT-AIL: Behavior Transformer-Assisted Adversarial Imitation
Learning from Human Gameplay in Gran Turismo Sport." Website: [link](https://sites.google.com/berkeley.edu/bet-ail/home), Paper link forthecoming.

### Code availability:
This repository contains pseudocode detailing the training process and BeT-AIL algorithm as a supplement to our original manuscript. The agent interface in Gran Turismo Sport is not enabled in commercial versions of the game. Please refer to [this article](https://rdcu.be/dssHS) for more information.

In this repository, we provide an implementation in the "MountainCarContinuous-v0" environment. The implementation is identical to the results presented in the manuscript in the Gran Turismo Sport environment. The hyperparameters are set to those used in the GTS experiments in the paper, and have not been optimized for the Mountain Car environment. 

## Usage

We provide a requirements.txt file with the required packages which can be installed with 
`pip install -r requirements.txt`

To train a BeT-AIL policy, run:
`python main.py --algorithm=BeT-AIL`

To train an AIL policy, run: 
`python main.py --algorithm=AIL`

To train an BeT policy, run: 
`python main.py --algorithm=BeT`

## References
Our AIL implementation is based on the pypi imitation library available here: [imitation](https://github.com/HumanCompatibleAI/imitation). Our Behavior Transformer implementation is based on the Decision Transformer implementation available here: [online-dt](https://github.com/facebookresearch/online-dt). Please cite the respective authors if you employ their code in your research.

