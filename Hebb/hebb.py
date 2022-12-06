import numpy as np
import matplotlib.pyplot as plt 

myfile = np.loadtxt('data/4.txt')
x_i = myfile[:, :2]
y_out = myfile[:, 2].reshape(len(myfile), 1)
ones = np.ones(len(myfile))
cell_inputs = np.column_stack((x_i, ones))
w_old = np.zeros((len(myfile), 3))
w_new = np.zeros((len(myfile), 3))
xi_y = cell_inputs * y_out

for i in range(len(myfile)):
  for j in range(3):
    if i == 0:
      w_new[i, j] =xi_y[i, j]
    else:
      w_new[i, j] =xi_y[i, j] + w_old[i-1, j]
    w_old[i, j] = w_new[i, j]
w1_new = w_new[len(myfile)-1,0]
w2_new = w_new[len(myfile)-1,1]
b_new = w_new[len(myfile)-1,2]

print('w_new', w_new[len(myfile)-1:, :])

x = myfile[:, :1]
y = myfile[:, 1:2]
lable = np.zeros((len(myfile)))
for i in range(len(myfile)):
  if y_out[i] < 0:
    lable[i] = int(2)
  else: 
    lable[i] = int(1) 
plt.scatter(x, y, c=lable)
plt.axis([-1, 1, -1, 1])
plt.grid()
x1 = np.linspace(-1, 1, 10)
x2 = (-w2_new * x1 - b_new) / w1_new

plt.plot(x2, x1)
plt.title("Hebb's training algorithm")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()