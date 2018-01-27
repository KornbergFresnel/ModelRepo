# CommNet
an implementation of CommNet, arxiv link: [Learning Multiagent Communication with Backpropagation](http://arxiv.org/abs/1605.07736)

**What's CommNet**

Many tasks in AI require the collaboration of multiple agents. Typically, the communication protocol between agents is manually specified and not altered during training. *CommNet* is a simple neural model, that uses continuous communication for fully cooperative tasks. The model consists of multiple agents and the communication between them is learned alongside their policy. 

*Demonstrating*

demonstrating the ability of the agents to learn to communicate amongst themselves, yielding improved performance over non-communicative agents and baselines. In some cases, it is possible to interpret the language devised by the agents, revealing simple but effective strategies for solving the task at hand.

*Structure*

<img src="./assets/commnet_model.png" width="70%" />

*Tasks*

1. Leaver Pulling game
2. Traffic Junction

**Plot of Loss (leaver pulling game)**

<img src="./assets/loss.png" width="70%" />

**Plot of reward and baseline**

<img src="./assets/reward.png" width="70%"/>
