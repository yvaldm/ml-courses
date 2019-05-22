import numpy as np
from urllib.request import urlopen

response = urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')
txt = np.loadtxt(response, skiprows=1, delimiter=',')

a = np.mean(txt, axis=0)
print(a)
