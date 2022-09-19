# name:shuaiqiuping
# university:sichuan normal unniversity
# time:2022/4/22 10:24

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

print(torch.__version__)

#数据准备

#对数据做标准化，将数据归一化到-1到1之间
transform=transforms.Compose([
    transforms.ToTensor(),  #0-1:channel,h,w
    transforms.Normalize(0.5,0.5) #(-1,1)
])
train_ds=torchvision.datasets.MNIST('data',train=True,transform=transform,download=True)
dataloader=torch.utils.data.DataLoader(train_ds,batch_size=64,shuffle=True)
imgs,_=next(iter(dataloader))
print(imgs.shape)#从loader中提取出一打数据来看看它是什么shape[64,1,28,28]

#定义生成器

#生成器的输入是长度为100的噪声（正态分布随机数）
#生成器的输出是图片[1,28,28]
"""
          Linear 1: 100-->256
          Linear 2: 256-->512
          Linear 2: 512-->28*28
          reshape: 28*28-->1,28,28
        """
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main=nn.Sequential(
             nn.Linear(100,256),
             nn.ReLU(),
             nn.Linear(256,512),
             nn.ReLU(),
             nn.Linear(512,28*28),
             #生成器模型的最后一层的激活函数要用tanh进行激活，经验只谈
             nn.Tanh()#-1 ,1
         )

    def forward(self,x):#x表示噪声为100的输入
         x=self.main(x)
         x=x.view(-1,28,28)#这里channel放在最后，方便后面绘图，这是reshape过程
         return x


#定义判别器模型
#判别器输入为（1，28，28）的图片，输出为二分类概率值，输出使用sigmoid激活0-1
#BCEloss来计算二分类交叉熵损失
#LeakyReLu():x>0,输出x，x<0,输出a*x，a表示一个很小的斜率，比如0.1，判别器中一般建议使用LeakyRelu
class Discrimination(nn.Module):
    def __init__(self):
        super(Discrimination, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),#相对于Relu函数，在小于0的部分会保留一定的梯度
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),#代表概率输出
            # 生成器模型的最后一层的激活函数要用tanh进行激活，经验只谈
            nn.Sigmoid()
        )

    def forward(self,x):
        x=x.view(-1,28*28)
        x=self.main(x)
        return x





#初始化模型、、优化器及损失计算函数
device='cuda' if torch.cuda.is_available() else 'cpu'
gen=Generator().to(device)
dis=Discrimination().to(device)
#定义判别器和生成器的优化器
d_optim=torch.optim.Adam(dis.parameters(),lr=0.0001)
g_optim=torch.optim.Adam(gen.parameters(),lr=0.0001)
#定义损失函数
loss_fn=torch.nn.BCELoss()#二分类使用交叉熵，用sigmod函数激活时用这个


#绘图函数，方便我们将每个批次当中生成器生成的图片绘制出，看看怎么样，生成的图片经过训练从生成一张张模糊的图片到生成清晰的数字图片
def gen_img_plot(model,test_input):#生成16张图片
    prediction=model(test_input).detach().cpu().numpy()
    prediction=np.squeeze(prediction)
    fig=plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    plt.show()

test_input=torch.randn(16,100,device=device)#这个是设置16个长度为100的随机噪声




#GAN的训练
D_loss=[]
G_loss=[]#这是为了把每个epoch的生成器loss和判别器loss记录下来
for epoch in range(5):
    d_epoch_loss=0
    g_epoch_loss=0
    count=len(dataloader)#一共多少个批次
    for step,(img,_)in enumerate(dataloader):
        img=img.to(device)#这里是一个batch的图片，64张
        size=img.size(0)#取上面得到的img的shape的第一维，也就是64
        random_noise=torch.randn(size,100,device=device)#生成64个随机的长度为100的噪声
        d_optim.zero_grad()#清空判别器的梯度
        real_output=dis(img)#得到真实图片的判别器判断概率
        d_real_loss=loss_fn(real_output,
                            torch.ones_like(real_output)
                            )#计算判别器的loss，torch.ones_like(real_output)相当于label
        d_real_loss.backward()
        gen_img=gen(random_noise)#生成器生成图片
        fake_output=dis(gen_img.detach())#把生成图片放进判别器判别,因为
        d_fake_loss=loss_fn(fake_output,
                            torch.zeros_like(fake_output)
                            )#对于判别器来讲它希望能够对生成的图片判别为假
        d_fake_loss.backward()
        d_loss=d_real_loss+d_fake_loss
        d_optim.step()#判别器的优化器step（）一下

        #上面是判别器的优化，包括两个部分，要把真的图片判断为真，要把假的图片判断为假

        #下面是生成器的优化，只包括一部分，希望判别器能把假图片判别为真
        g_optim.zero_grad()
        fake_output=dis(gen_img)
        g_loss=loss_fn(fake_output,torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss+=d_loss#把每次循环 中的loss都加上
            g_epoch_loss+=g_loss

    with torch.no_grad():
        #计算每个epoch的平均d_loss和平均g_loss
        d_epoch_loss/=count
        g_epoch_loss/=count
        #加到这个最初设置的loss列表中
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:',epoch)
        gen_img_plot(gen,test_input)