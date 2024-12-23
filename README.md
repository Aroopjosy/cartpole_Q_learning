# Gymnasium Cartpole

This repository contains an implementation to solve the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) problem from the OpenAI Gym or Gymnasium . The goal is to balance a pole on a moving cart by applying discrete forces.

## Demo Video
![Watch the demo video](https://github.com/Aroopjosy/cartpole_balance/blob/main/demo.gif)



## Table Content

- [Introduction](https://github.com/Aroopjosy/cartpole_balance?tab=readme-ov-file#introduction)
- [Demo Video](https://github.com/Aroopjosy/cartpole_balance?tab=readme-ov-file#demo-video-1)
- [How to Run the Code](https://github.com/Aroopjosy/cartpole_balance?tab=readme-ov-file#how-to-run-the-code)

## Introduction

The CartPole problem is a classic reinforcement learning challenge where an agent learns to balance a pole on a moving cart. The goal is to apply forces to keep the pole upright as long as possible. In this project, we solve the problem using [Q-learning](https://en.wikipedia.org/wiki/Q-learning), a fundamental algorithm in reinforcement learning.




## Demo Video
![Watch the demo video](https://github.com/Aroopjosy/cartpole_balance/blob/main/demo.gif)


## How to Run the Code

1. clone & install depedencies
   ```bash
    git clone https://github.com/Aroopjosy/cartpole_balance.git
    cd cartpole_balance
    pip install -r requirements.txt
   ```
2. create virual environment
    ```bash
    virtualenv cartpole-env
    source cartpole-env/bin/activate
    ```

3. Train 
    ``` bash 
    python train_cartpole.py
    ```
    _Only run after training completed._
4. Test
    ```bash
    python test_cartpole.py
    ```


