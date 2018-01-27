# ModelRepo
> reproduce some RL or Multi-Agent models

**[NFSP (Neural Fictitious Self-Play)](https://github.com/KornbergFresnel/ModelRepo/tree/master/NFSP)**

> NFSP is a kind of training method for improving the performance of deep networks used in RL tasks, you can read more details in 👇 

- arxiv link: [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](http://arxiv.org/abs/1603.01121)

- the work of implemenation is still in process ...

**[Multi-Agent Deep Deterministic Policy Gradient](https://github.com/KornbergFresnel/ModelRepo/tree/master/MADDPG)**

> This algorithm is proposed by *Ryan Lowe* and *Yi Wu* at 2017, it is a multi-agent actor-critic framework essentially, you can read more by visiting 👇 arxiv link. My implementation supports several environments but not all of MA environments so far

- arxiv link: [Multi-Agent Actor-Critic Mixed Cooperative-Competitive Environments 2017 NIPS](https://arxiv.org/abs/1706.02275)

- run `run.py` do traninig task, get more information about executation: `python run.py -h`

- if you wanna try different parameters configuration, you can modify the `config.py`


**[CommNet: Learning Multiagent Communication with Backpropagation](https://github.com/KornbergFresnel/ModelRepo/tree/master/CommNet)**

> CommNet was proposed by *Sainbayar Sukhbaatar*, *Arthur Szlam* and *Rob Fergus* at 2016, it has a simple implementation of multi-agent communication with deep network. They tested algorithm under different playground, and experiments showed that CommNet had a better performance than naked implementation without communication

- arxiv link: [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736)

- run `leaver_train.py` to do training task, and you can also visit [KornbergFrsnel: CommNet](https://github.com/KornbergFresnel/CommNet) directly to get more information

- the paper has several different playgrounds, while there has only *Leaver* implemented in this repo so far (maybe I will add more playgrouds, but who knows.)
