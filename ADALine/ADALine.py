import numpy as np
import time, random, math
import matplotlib.pyplot as plt

x1, x2, target = np.loadtxt('data/1.txt', delimiter='\t', unpack=True)
xi = np.column_stack((x1, x2, np.ones(len(target))))
alpha = 0.08; Epsilon = 0.00005    # <------ Set alpha & Epsilon
stop_condition = 1; Epoch = 0; last_error = 0; error = 0
wi = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]  # <------ Initialize weights and bias
lable = np.zeros((len(target)))   # <------ Draw Charts
for i in range(len(target)):
  if target[i] > 0:
    lable[i] = int(1)
  else:
    lable[i] = int(2)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(x1, x2, c=lable)
plt.grid(); plt.xlabel('X'); plt.ylabel('Y')
x = np.linspace(-1, 1, 100)
y1 = np.linspace(-1, 1, 100)
# y2 = np.linspace(-1, 1, 100)
# y3 = np.linspace(-1, 1, 100)
line1, = ax.plot(x, y1)
# line2, = ax.plot(x, y2)
# line3, = ax.plot(x, y3)
while stop_condition:
  last_error = error
  error = 0
  for i in range(len(target)):
    y_in = (xi[i] * wi).sum()
    delta_wi = alpha * (target[i] - y_in) * xi[i]
    wi += delta_wi
    error += math.pow((target[i] - y_in), 2) / 2
  plt.title(f"ADALine training algorithm (1), Epoch = {Epoch} \n Error_changes = {abs(error - last_error)} \n wi = {wi}")
  plt.scatter(x1, x2, c=lable)    #
  new_y1 = (-wi[0] * x - wi[2]) / wi[1]
  # new_y2 = (-w1 * x - (b + tetha)) / w2
  # new_y3 = (-w1 * x - b) / w2
  line1.set_xdata(x)    # updating the value of x & y
  line1.set_ydata(new_y1)
  # line2.set_xdata(x)
  # line2.set_ydata(new_y2)
  # line3.set_xdata(x)
  # line3.set_ydata(new_y3)
  fig.canvas.draw()   # re-drawing the figure
  fig.canvas.flush_events()   # to clear the GUI events
  plt.savefig('ADALine.png')
  time.sleep(0.1)
  Epoch += 1
  if abs(error - last_error) < Epsilon:
    stop_condition = 0
    print('Epoch',Epoch,'=', wi)      