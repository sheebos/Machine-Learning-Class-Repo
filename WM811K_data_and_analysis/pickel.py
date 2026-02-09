# loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline 

import os
print(os.listdir("X:/ECEsite/Machine-Learning-Class-Repo/"))
import warnings
warnings.filterwarnings("ignore")

df = pd.read_pickle(
    "X:/ECEsite/Machine-Learning-Class-Repo/LSWMD.pkl"
)
df.info()
