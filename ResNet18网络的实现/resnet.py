# name:shuaiqiuping
# university:sichuan normal unniversity
# time:2022/4/22 10:24


import torch
from torch import nn
from torch.nn import  functional as F


class ResBlk(nn.Module):#残差网络结构
    #残差网络初始化要知道输入和输出
    def __init__(self,ch_in,ch_out):
        super(ResBlk, self).__init__()

        #这里是一个残差网络结构中所有的网络单元
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1 )
        self.bn2=nn.BatchNorm2d(ch_out)
        self.extra=nn.Sequential()#设置一个空的网络单元，如果残差网络结构的输入和输出不一样，就矫正一下
        if ch_out != ch_in:
            # [b,ch_in,h,w]=>[b,ch_out,h,w]
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self,x):#将所有的网络单元连起来构成一个前向传播的网络
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out=self.extra(x)+out #shortcut
        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()


        #ResNet18进去首先是一个卷积网络,3个通道的图片，变成64个通道
        self.conv1=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32)
        )
        #然后后面跟了4个残差结构
        #[b,64,h,w]=>[b,128,h,w]
        self.blk1=ResBlk(32,64)
        # [b,64,h,w]=>[b,128,h,w]
        self.blk2 = ResBlk(64, 128)
        # [b,128,h,w]=>[b,256,h,w]
        self.blk3 = ResBlk(128, 256)
        # [b,256,h,w]=>[b,512,h,w]
        self.blk4 = ResBlk(256, 512)

        #最后用一个线性层给它转化成10类
        self.outlayer=nn.Linear(512*32*32,10)


    def forward(self,x):
        x=F.relu(self.conv1(x))
        #[b,64,h,w]=>[b,1024,h,w]
        x=self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        #转化线性层要打平
        x=x.view(x.size(0),-1)
        x=self.outlayer(x)
        return x


def main():
    blk=ResBlk(64,128)#测试ResBlk,希望输入64个channel，输出128个channel
    tmp=torch.randn(2,64,32,32)
    out=blk(tmp)
    print('block output:',out.shape) #torch.Size([2, 128, 32, 32])测试正确
    Net=ResNet18()
    tmp=torch.randn(2,3,32,32)#2张3通道照片传到ResNet18网络里面
    out=Net(tmp)
    print('ResNet18 output:', out.shape)


if __name__ == '__main__':
    main()
