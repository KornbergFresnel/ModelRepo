import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


filename = sys.argv[1]
data = [[], []]


with open(filename, 'r') as f:
    i = 0
    for line in f.readlines():
        items = line.split("\t")
        row = []
        
        # [loss, reward]
        for item in items[1:]:
            t = eval(item.split(':')[1])
            row.append(t)

        if len(row) > 0:
            data[i % 2].append(row)
        i += 1


# actor: loss, reward
data[0] = np.array(data[0])
# critic: loss, baseline
data[1] = np.array(data[1])


matplotlib.style.use("ggplot")


plt.figure(figsize=(15, 10))
fig1 = plt.subplot(211)
fig2 = plt.subplot(212)


if sys.argv[2] == "plotloss":

    fig1.plot(np.arange(len(data[0])), data[0][:, 0])
    fig1.set_ylabel("Loss value")
    fig1.set_title("Loss per 10 episodes")

    fig2.plot(np.arange(len(data[1])), data[1][:, 0])
    fig2.set_xlabel("Episodes per 10")
    fig2.set_ylabel("Loss value")

elif sys.argv[2] == "plotvalue":
    fig1.plot(np.arange(len(data[0])), data[0][:, 1])
    fig1.set_ylabel("Rate value")
    fig1.set_title("Reward per 10 episodes")

    fig2.plot(data[1][:, 1])
    fig2.set_xlabel("Episodes per 10")
    fig2.set_ylabel("Baseline value")
    fig2.set_title("Baseline per 10 episodes")

plt.show()




