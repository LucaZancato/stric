import numpy as np

def create_Hankel(data, win_lenght, n):
    Hankel = []
    for i in range(len(data)-win_lenght):
        Hankel.append(data[i:i+win_lenght].reshape(-1,1))
    past, future = np.concatenate(Hankel[:n], 1).T, np.concatenate(Hankel[n:2*n], 1).T
    return past, future
