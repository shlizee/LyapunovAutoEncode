import matplotlib.pyplot as plt
import numpy as np

def display(ft, simtime=None, title=None, color=None, lineWidth=100):
    if simtime is None:
        simtime = range(0, len(ft))
    if color is None:
        plt.plot(simtime, ft, linewidth=lineWidth)
    else:
        plt.plot(simtime, ft, color, linewidth=lineWidth)
    plt.title(title)
