import math
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
 

class Tanh:       
    def forward(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def backward(self, outx):
        tanh = (np.exp(outx) - np.exp(-outx)) / (np.exp(outx) + np.exp(-outx))
        return 1 - math.pow(tanh, 2)
 

def Axis(fig, ax):
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["x"].set_axisline_style("->", size = 1.0)
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("->", size = 1.0)
    ax.axis["y"].set_axis_direction("right")
 
x =  np.linspace(-10, 10, 100)
y_forward = []
y_backward = []
 
def get_list_forward(x):
    for i in range(len(x)):
        y_forward.append(Tanh().forward(x[i]))
    return y_forward
 
def get_list_backward(x):
    for i in range(len(x)):
        y_backward.append(Tanh().backward(x[i]))
    return y_backward
    
y_forward = get_list_forward(x)
y_backward = get_list_backward(x)

fig = plt.figure(figsize=(12, 12))

ax = axisartist.Subplot(fig, 111)
Axis(fig, ax)

plt.ylim((-2, 2))
plt.xlim((-10, 10))

plt.plot(x, y_forward, color='red', label='$f(x) = tanh(x)$')
plt.legend()

plt.plot(x, y_backward, label='f(x)\' = 1-(tanh)^2')
plt.legend()
 
plt.show()