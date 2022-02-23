from cgi import print_form
import numpy as np
import matplotlib.pyplot as plt
import os

dataSet = np.load('./DataSet/data_test.npy')

print(len(dataSet[0][0]))
