import matplotlib.pyplot as plt
import numpy as np


with open('./train_accuracy.txt') as ta:
    train_accuracy = ta.readlines()
train_accuracy = [float(x) for x in train_accuracy]

with open('./val_accuracy.txt') as va:
    val_accuracy = va.readlines()
val_accuracy = [float(x) for x in val_accuracy]

with open('./val_loss.txt') as vl:
    val_loss = vl.readlines()
val_loss = [float(x) for x in val_loss]

with open('./train_loss.txt') as tl:
    train_loss = tl.readlines()
train_loss = [float(x) for x in train_loss]

with open('./loss_diff.txt') as ld:
    loss_diff = ld.readlines()
loss_diff = [float(x) for x in train_loss]

x1 = np.arange(0, len(train_accuracy) )
x2 = np.arange(0, len(val_accuracy) )
x3 = np.arange(0, len(train_loss) )
x4 = np.arange(0, len(val_loss))
x5 = np.arange(0, len(loss_diff))


fig, axs = plt.subplots(5)
axs[0].plot(x1, train_accuracy)
axs[0].set_title('train accuracy')
axs[1].plot(x2, val_accuracy)
axs[1].set_title('validation accuracy')
axs[2].plot(x3, train_loss)
axs[2].set_title('train loss')
axs[3].plot(x4, val_loss)
axs[3].set_title('val loss')
axs[4].plot(x5, loss_diff)
axs[4].set_title('loss difference')
plt.show()