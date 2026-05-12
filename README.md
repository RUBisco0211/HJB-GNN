# HJB-GNN
Jax Official Implementation of **RSS 2026** Paper: [F. Wang](https://scholar.google.com/citations?user=wo6W6DUAAAAJ&hl=en&oi=sra), [X. Shu](https://scholar.google.com/citations?user=B35tin4AAAAJ&hl=en&oi=sra), [L. He](https://scholar.google.com/citations?user=QGwYalkAAAAJ&hl=en&oi=sra), [L. Zhao*](https://sites.google.com/view/lzhao?pli=1): "[Safe Multi-Agent Navigation via Constrained HJB-Informed Learning](https://nus-core.github.io/assets/standalone/HJB-GNN/index.html)". 

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n hjbgnn python=3.10
conda activate hjbgnn
cd hjbgnn
```

Then install jax following the [official instructions](https://github.com/google/jax#installation), and then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

## Installation

Install: 

```bash
pip install -e .
```

## Run

### Environments

We provide the `CrazyFlie` environment.

### Algorithms

We provide the HJB-GNN algorithm (`hjb_gnn`). Use `--algo` to specify the algorithm.

### Hyper-parameters

To reproduce the CrazyFlie results shown in our paper, one can refer to [`settings.yaml`](./settings.yaml).

### Train

To train the model, use:

```bash
python train.py --algo hjb_gnn --env CrazyFlie -n 8 --area-size 2 --n-env-train 16 --loss-action-coef 1e-4 --loss-value-coef 1e-4
```

In our paper, we use 8 agents with 1000 training steps. The training logs will be saved in folder `./logs/<env>/<algo>/seed<seed>_<training-start-time>`. We also provide the following flags:

- `-n`: number of agents
- `--env`: environment (currently `CrazyFlie`)
- `--algo`: algorithm, including `hjb_gnn`
- `--seed`: random seed
- `--steps`: number of training steps
- `--name`: name of the experiment
- `--debug`: debug mode: no recording, no saving
- `--obs`: number of obstacles
- `--n-rays`: number of LiDAR rays
- `--area-size`: side length of the environment
- `--n-env-train`: number of environments for training
- `--n-env-test`: number of environments for testing
- `--log-dir`: path to save the training logs
- `--eval-interval`: interval of evaluation
- `--eval-epi`: number of episodes for evaluation
- `--save-interval`: interval of saving the model

In addition, use the following flags to specify the hyper-parameters:
- `--alpha`: GCBF alpha
- `--horizon`: look forward horizon
- `--lr-actor`: learning rate of the actor
- `--lr-cbf`: learning rate of the CBF
- `--loss-action-coef`: coefficient of the action loss
- `--loss-value-coef`: coefficient of the value loss
- `--loss-h-dot-coef`: coefficient of the h_dot loss
- `--loss-safe-coef`: coefficient of the safe loss
- `--loss-unsafe-coef`: coefficient of the unsafe loss
- `--buffer-size`: size of the replay buffer

### Test

To test the learned model, use:

```bash
python test.py --path <path-to-log> --epi 5 --area-size 2 -n 8 --obs 0
```

This should report the safety rate, goal reaching rate, and success rate of the learned model, and generate videos of the learned model in `<path-to-log>/videos`. Use the following flags to customize the test:

- `-n`: number of agents
- `--obs`: number of obstacles
- `--area-size`: side length of the environment
- `--max-step`: maximum number of steps for each episode, increase this if you have a large environment
- `--path`: path to the log folder
- `--n-rays`: number of LiDAR rays
- `--alpha`: CBF alpha
- `--max-travel`: maximum travel distance of agents
- `--cbf`: plot the CBF contour of this agent, only support 2D environments
- `--seed`: random seed
- `--debug`: debug mode
- `--cpu`: use CPU
- `--u-ref`: test the nominal controller
- `--env`: test environment (not needed if the log folder is specified)
- `--algo`: test algorithm (not needed if the log folder is specified)
- `--step`: test step (not needed if testing the last saved model)
- `--epi`: number of episodes to test
- `--offset`: offset of the random seeds
- `--no-video`: do not generate videos
- `--log`: log the results to a file
- `--dpi`: dpi of the video
- `--nojit-rollout`: do not use jit to speed up the rollout, used for large-scale tests

### Pre-trained models

We provide the pre-trained models in the folder [`pretrained`](pretrained).
