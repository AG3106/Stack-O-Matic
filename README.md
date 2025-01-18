# Stack-O-Matic
Assignment Solutions for Stack-O-Matic winter project

**Assignment 1**

Install Required Libraries

```bash
$ pip install numpy
$ pip install pandas
```

Run assignment1.ipynb

**Assignment 2**

Install Required Libraries

```bash
$ pip install numpy
$ pip install pandas
$ pip install matplotlib
```

Run assignment2.ipynb

**Assignment 3**

Install Required Libraries
```bash
$ pip install matplotlib
$ pip install torch
$ pip install swig
$ pip install gymnasium[box2d]
```
If `pip install gymnasium[box2d]` gives error in building. Try
```bash
$ pip install wheel setuptools pip --upgrade
$ pip install swig
$ pip install gymnasium[box2d]
```

Run assignment3.ipynb

**Assignment 4**

Install Required Libraries
```bash
$ pip install matplotlib
$ pip install torch
$ pip install pygame
$ pip install numpy
$ pip install moviepy==1.0.3
```

`model.py` contains DQN model architecture

`dqn_agent_training.py` is for training model

`tetris_env.py` has tetris game environment

`pygameview.py` for generating pygame window of game and saving video

`test.py` for testing trained model