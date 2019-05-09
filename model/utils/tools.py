# -*- coding:utf-8 -*-

import numpy as np

c = np.random.rand(1000,2)
a = np.random.randint(5,10,(480,30))
b = a.reshape((16,30,30))

print(b)