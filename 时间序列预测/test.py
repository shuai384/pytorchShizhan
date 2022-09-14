# name:shuaiqiuping
# university:sichuan normal unniversity
# time:2022/4/22 10:24
import torch
import  torch.nn as nn
import  torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

num_time_steps=50#一段波形里面50个点
input_size=1#相当于有50个单词，1句话，每个单词用一个实数表示【50，1，1】
hidden_size=16#memory维度定义为16，最终输出的out维度也一定是16，整合起来【50，1，16】
output_size=1
lr=0.01

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()

        self.rnn=nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True#因为batch放在第一个维度
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p,mean=0.0,std=0.001)

        #用一个线性层把16维变成1维
        self.linear=nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden_prev):
        # x:[1,50,1]
        #hidden_prev:[b,16]
        out,hidden_prev=self.rnn(x,hidden_prev)#out[b,50,16]h[b,16]
        out=out.view(-1,hidden_size)#将out打平送到线性层[b,50*16]
        out =self.linear(out)#[b,1]
        out=out.unsqueeze(dim=0)#在第0维的位置上添加1维[1,seq,1],这里为什么要用三个维度，因为这是从网络中出去的预测值，要和y做一个均方差计算loss
        return  out,hidden_prev


# x=torch.randn(1,50,1)
# h=torch.zeros(1,1,16)
# model=Net()
# output,hidden_prev=model(x,h)
# print(output.shape,hidden_prev.shape)
#torch.Size([1, 50, 1]) torch.Size([1, 1, 16])


model=Net()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr)

hidden_prev=torch.zeros(1,1,hidden_size)#将hidden_prev初始化

for iter in range(6000):
    start=np.random.randint(3,size=1)[0]#随便搞一个3以内的数作为起始点
    time_steps=np.linspace(start,start+10,num_time_steps)#在起始点->起始点+10这段数据内取50个点
    data=np.sin(time_steps)#[1,50]
    data=data.reshape(num_time_steps,1)#[50,1]
    x=torch.tensor(data[:-1]).float().view(1,num_time_steps-1,1)#0~48 [1,49,1]
    y=torch.tensor(data[1:]).float().view(1,num_time_steps-1,1)#1~49 [1,49,1]
    #(x,y)是一对数据对

    output,hidden_prev=model(x,hidden_prev)#[b,50,1][b,1,16]
    hidden_prev=hidden_prev.detach()#detach的作用是深拷贝解决tenso.date()的安全性

    loss=criterion(output,y)
    model.zero_grad()
    loss.backward()

    optimizer.step()

    if iter%100==0:
        print("第{}次循环:loss:{}".format(iter,loss.item()))


#下面是做预测

#生成数据
start=np.random.randint(3,size=1)[0]
time_steps=np.linspace(start,start+10,num_time_steps)
data=np.sin(time_steps)
data=data.reshape(num_time_steps,1)
x=torch.tensor(data[:-1]).float().view(1,num_time_steps-1,1)
y=torch.tensor(data[1:]).float().view(1,num_time_steps-1,1)

#做预测，前面的代码训练好了以后，我们随便生成一个数据x，用它来预测下一个数据，再用这个数据预测下一个数据，循环，这样我们就可以得出一条预测曲线
predictions=[]
input=x[:,0,:] #x[1,49,1]=>[1,1]
for _ in range(x.shape[1]):#从0到48
    input =input.view(1,1,1)#[1,1]=>[1,1,1]
    (pred,hidden_prev)=model(input,hidden_prev)
    input=pred
    predictions.append(pred.detach().numpy())



#画图

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], predictions)
plt.show()



