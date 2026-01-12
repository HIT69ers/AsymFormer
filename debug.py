import numpy as np

palette = np.load('nyucmap.npy')


for i in range(41):
    print(f"{i}——{palette[i]}")