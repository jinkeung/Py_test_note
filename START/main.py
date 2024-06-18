import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

iris=load_iris()
data=iris.data
field=iris.feature_names
index=iris.target

df=pd.DataFrame(data,index,field)
print(df)

for i in range(0,3):
    plt.scatter(data[index==i,0],data[index==i,1])
plt.show()