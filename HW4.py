# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:52:02 2015

@author: DAN
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


prob1Data = None



def prob1():
    global prob1Data
    prob1Data = pd.DataFrame([(6, 12), (19, 7), (15, 4), (11, 0), 
                              (18, 12), (9, 20), (19, 22), (18, 17), 
                            (5, 11), (4, 18), (7, 15), (21, 18), (1, 19), 
                            (1, 4), (0, 9), (5, 11)], columns=['X','Y'])

    plt.scatter(prob1Data.X, prob1Data.Y)
    plt.show()


def main():
    print("In Main.")
    prob1()





if __name__ == "__main__":
    main()

