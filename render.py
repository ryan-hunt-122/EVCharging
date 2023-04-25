import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import sys

with open(f'pickles/0.0001_10.0/ep{sys.argv[1]}.pkl', 'rb') as fp:
    data = pickle.load(fp)

# create the figure and axes objects
fig, ax = plt.subplots(nrows=2, ncols=1)

ref = []
act = []
tot = 0

for d in data:
    a, b = d
    ref.append(b)
    act.append(sum(a['dt_s']))
    tot += b - sum(a['dt_s'])

def animate(i):
    d, _ = data[i]
    y = d['t_s']
    x = d['t_d']

    ax[0].clear()
    ax[0].scatter(x, y)
    ax[0].plot(range(300), range(300))
    # ax.imshow(d, cmap='hot', interpolation='nearest')
    ax[0].set_xlim([0, 10])
    ax[0].set_ylim([0, 10])
    for j in range(len(x)):
        ax[0].arrow(x[j], y[j], -d['dt_d'][j]*0.05, -d['dt_s'][j]*0.05, width=0.05)
    ax[0].grid()

    ax[1].clear()
    ax[1].plot(range(len(ref)), ref, label='Reference')
    ax[1].plot(range(len(act)), act, label='Action Sum')
    ax[1].axvline(i)
    ax[1].legend()

print(tot)
ani = FuncAnimation(fig, animate, frames=4, interval=500, repeat=True)

plt.show()
