import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
imoprt matplotlib.pyplot as plt

training_dataset=paddle.vision.datasets.MNIST(model='train')
train_data0=np.array(train_dataset[0][0])
train_label_0=np.array(train_dataset[0][1])

import matplotlib.pyplot as plt
plt.figure("Image")
plt.figure(figsize=(2,2))
plt.imshow(train_data0,cmap=plt.cm.binary)
plt.axis('on')
plt.title('image')
plt.show()

print("图像数据形状和对应数据为：",train_data0.shape)
print("图像标签形状和对应数据为：",train_label_0.shape,train_label_0)
print("\n打印第一个batch的第一个图像，对应的标签数字为{}'.format(train_label_0))

class MNIST(paddle.nn.layer):
    def _init_(self):
        super(MNIST,self).init_()
        self.fc=paddle.nn.Linear(in_features=784,out_features=1)
    def forward(self,inputs):
        outputs=self.fc(inputs)
        return outputs

model=MNIST()

def train(model):
    model.train()
    train_loader=paddle.io.DataLoader(paddle.vision.dataset.MNIST(model='train'),
                                        batch_size=16,
                                        shuffle=True)
    opt=paddle.optimizer.SGD(learning_rate=0.001,parameters=model.parameters())

def norm_img(img):
    assert len(img.shape_==3
    batch_size,img_h,img_w=img.shape[0],img.shape[1],img.shape[2]
    img=img/255
    img=paddle.reshape(img,[batch_size,img_h*img_w])
    return img

import paddle
paddle.vision.set_image_backend('cv2')
model=MNIST
def train(model):
    model.train()
    train_loader=paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                     batch_size=16,
                                     shuffle=True)
    opt=paddle.optimizer.SGD(learning_rate=0.001,parameters=model.parameters()0
    EPOCH_NUM=10
    for epoch in range（EPOCH_NUM）:
        for batch_id,data in enumerate(train_loader()):
            images=norm_img(data[0]).astype('float32')
            labels=data[1].astype('float32')
            predicts=model(images)
            loss=F.square_error_cost(predicts,labels)
            avg_loss=paddle.mean(loss)
            if batch_id%1000==0:
                print("epoch_id:{},batch_id:{},loss is:{}".format(epoch,batch_id,avg_loss.numpy()))

            avg_loss.backward()
            opt.step()
            opt.clear_grad()

train(model)
paddle.save(model.state_dict(),'./mnist.pdparams')
