import numpy as np
import time
import matplotlib.pyplot as plt 

x1, x2, target = np.loadtxt('data/4.txt', delimiter='\t', unpack=True)
alpha = 1; tetha = 1    # <------ Set alpha & tetha
stop_condition = 1; Epoch = 0    
w1 = 0; w2 = 0; b = 0   # <------ Initialize weights and bias

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
x = np.linspace(-1, 1, 100)
y1 = np.linspace(-1, 1, 100)
y2 = np.linspace(-1, 1, 100)
y3 = np.linspace(-1, 1, 100)
line1, = ax.plot(x, y1)
line2, = ax.plot(x, y2)
line3, = ax.plot(x, y3)

while stop_condition:
  y = np.zeros((len(target), 1))
  count_w_change = 0
  for i in range(len(target)):
    y_in = b + x1[i] * w1 + x2[i] * w2
    if y_in > tetha:    # <------ Two-threshold activation function
      y[i] = 1
    elif y_in < -tetha:
      y[i] = -1
    else:
      y[i] = 0
    if y[i] != target[i]:   # <------ Update weights and bias 
      w1 += alpha * target[i] * x1[i]
      w2 += alpha * target[i] * x2[i]
      b += alpha * target[i]
      count_w_change += 1
  Epoch += 1

  plt.grid(); plt.xlabel('X'); plt.ylabel('Y')
  plt.title(f"Perceptron's training algorithm (4), Epoch = {Epoch}")
  plt.scatter(x1, x2, c=lable)    #
  new_y1 = (-w1 * x - (b - tetha)) / w2   
  new_y2 = (-w1 * x - (b + tetha)) / w2
  new_y3 = (-w1 * x - b) / w2
  line1.set_xdata(x)    # updating the value of x & y
  line1.set_ydata(new_y1)
  line2.set_xdata(x)
  line2.set_ydata(new_y2)
  line3.set_xdata(x)
  line3.set_ydata(new_y3)
  fig.canvas.draw()   # re-drawing the figure
  fig.canvas.flush_events()   # to clear the GUI events
  plt.savefig('perceptron.png')
  time.sleep(0.2)
  if count_w_change == 0:   # <------ Stop condition
    stop_condition = 0
print('W1=', round(w1, 2), 'W2=', round(w2, 2), 'b=',b)