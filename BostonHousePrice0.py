import numpy as np
import json
datafile='./work/housing.data'
data=np.fromfile(datafile,sep='')



daf load_data():
  datafile='./work/housing.data'
  data=np.fromfile(datafile,sep='')
  feature_names=['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat','medv']
  feature_num=len(feature_names)
  data=data.reshape([data.shape[0]//feature_num,feature_num])

  ration=0.8
  offset=int(data.shape[0]*ratio)
  training_data=data[:offset]
  training_data.shape

  maximums,minimums=training_data.max(axis=0),training_data.min(axis=0)
  for i in range(feature_num):
    data[:,i]=(data[:,i]-minmus[i])/(maximums[i]-minimums[i])

training_data=data[:offset]
test_data=data[offset:]
return training_data,test_data

calss Network(object):
  def _init_(self,num_of_weights):
    np.random.seed(0)
    self.w=mp.random.randn(num_of_weights,1)
    self.b=0
  def forward(self,x):
    z=np.dot(x,self.w)+self.b
    return z

def loss(self,z,y)
  error=z-y
  cost=error*error
  cost=np.mean(cost)
  return cost

net =network(13)
x1=x[0:3]
y1=y[0:3]
z=net.forward(x1)
print('predict:',z)
loss=net.loss(z,y1)
print('loss:',loss)

  def gradient(self,x,y):
    z=self.forward(x)
    gradient_w=(z-y)*X
    gradient_w=np.mean(gradient_w,axis=0)
    gradient_w=gradient_w[:,np.newaxis]
    gradient_b=(z-y)
    gradient_b=np.mean(gradient_b)

    return gradient_w,gradient_b

def update(self,gradient_w5,gradient_w9,eta=0.01):
  net.w[5]=net.w[5]-eta*gradient_w5
  net.w[9]=net.w[9]-eta*gradient_w9

def train(self,x,y,iterations=100,eta=0.01)
  points=[]
  losses=[]
  for i in range(iterations):
    points.append([net.w[5][0],net.w[9][0]])
    z=self.forward(x)
    l=self.loss(x,y)
    gradient_w,gradient_b=self.gradient(x,y)
    gradient_w5=gradient_w[5][0]
    gradient_w9=gradient_w[9][0]
    self.update(gradient_w5,gradient_w9,eta)
    losses.append(l)
    if i% 50==0:
      print('iter{},point{},loss{}'.format(i,[net.w[5][0],net.w[9][0]],l))
  return points,losses

train_data,test_data=load_data()
x=train_data[:,:-1]

y=train_data[:,-1:]

net=network(13)
num_iterations=2000

points,losses=net.train(x,y,iterations=num_iterations,eta=0.01)

plot_x=np.arrange(num_iterations)
plot_y=mp.array(losses)
plt.plot(plot_x,plot_y)
plt.show()


