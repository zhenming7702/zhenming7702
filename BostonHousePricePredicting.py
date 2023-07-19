# paddlepaddle 重写波士顿房价预测

import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random

def loadData():
        datafile='./work/housing.data'
        data=np.fromfile(datafile,seg='',dtype=np.float32)
        feture_names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',\
                'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
        feature_num=len(feature_names)
        data=data.reshape({data.shape[0]//feature_num,feature_num])

        ratio=0.8
        offset=int(data.shape[0]*ration)
        training_data=data[:offset]
        maximums,minimums=training_data.max(axis=0),training_data.min(axis=0)

        global max_values
        global min_values

        max_values=maximums
        min_values=minimums

        for i in range(feature_num):
                data[:,i]=(data[:,i]-min_values[i])/(maximums[i]-minimus[i])
        training_data=data[:offset]
        test_data=data[offset:]
        return training_data,test_data
training_data,test_data=load_data()
print(training_data.shape)
print(training_data[1,:1])

calss Regressor(paddle.nn.layer):
        def _init_(self):
                super(Regressor,self)._init_()
                self.fc=Linear(in_features=13,out_features=1)
        def forward(self,inputs):
                x=self.fc(inputs)
                return x

model = Regressor()
model.train()
training_data,test_data=load_data()
opt=paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())

  
