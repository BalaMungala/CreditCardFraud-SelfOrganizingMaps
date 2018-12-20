# Data Preprocessing
import pandas as pd
dataset=pd.read_csv("Credit_card_Applications.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(x)

# Training
from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

# visualization
from pylab import bone,pcolor,colorbar,plot,show
import matplotlib.pyplot as plt
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
plt.savefig("som.png")
show()  

# Finding the frauds
import  numpy as np
mappings = som.win_map(X)  
frauds=np.concatenate((mappings[(8,7)],mappings[(8,8)]),axis=0)
frauds=sc.inverse_transform(frauds)
