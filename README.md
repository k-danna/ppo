# ppo

implementation of Proximal Policy Optimization Algorithms, OpenAI, 2017 (arxiv:1707.06347v2)

run main.py to replicate results, edit params.py if you want - any discrete gym env is compatible (200 max score)

10000 training episodes (with weight-sampled actions) on the cartpole problem from OpenAI gym. The problem is solved ~1000 episodes in with a single size 1024 hidden layer

![alt text](ppo.png)

