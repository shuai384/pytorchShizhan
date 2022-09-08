# name:shuaiqiuping
# university:sichuan normal unniversity
# time:2022/4/22 10:24
import torch
from torch import  nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()#初始化父类

        #卷积单元
        self.conv_unit=nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        )

        #flatten，从卷积层到全连接层要有一个打平操作，用下面的方法得知卷积层输出是【32，16，5，5】
        #打平变成16*5*5=400


        #全连接单元
        self.fc_unit=nn.Sequential(
            #这里如果不知道卷积单元打平后有多少个in，就先暂时写个2，用下面的代码来计算卷积单元输出的shape
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

       #因为卷积单元输出的shape不知道，于是就无法进行展平操作
       # 假设随机生成一个数据集(2,3,32,32)，把它喂到卷积单元，看它出来是个什么shape
       # 原本cifar数据集一个batch是32张图片，但是图片经过卷积层和全连接层不影响图片张数，改变的是图片的通道和长宽
       # 所以这里随机生成的tensor就假设有两张图片，减少计算梁
        tmp=torch.randn(2,3,32,32)
        out=self.conv_unit(tmp)
        print('conv_out:',out.shape)#[2, 16, 5, 5]
        #卷积单元出来的一张图片是16*5*5，把它拉平变成400


        #use Cross Entropy Loss
        #self.criteon=nn.CrossEntropyLoss()


    def forward(self,x):
            """
            :param x: [b,3,32,32]
            :return:
            """
            batchsz=x.size(0)#x的size就是[b,3,32,32]，然后(0)就是b
            #[b,3,32,32]=>[b,16,5,5]
            x=self.conv_unit(x)
            x=x.view(batchsz,16*5*5)#打平
            logits=self.fc_unit(x)
            #[b,10]
            #把这个logits换算成概率，最终得到b个概率，然后去算loss
            #但是nn.CrossEntropyLoss()里面已经包括了softmax，所以就不用写这一步了
            #pred=F.softmax(logits,dim=1)
            #loss=self.criteon(logits,y)
            return logits

def main():
    net=Lenet5()
    tmp=torch.randn(2,3,32,32)
    out=net(tmp)
    print('lenet out',out.shape)

if __name__ == '__main__':
    main()