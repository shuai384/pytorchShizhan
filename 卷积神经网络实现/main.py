import torch
from torch.utils.data import  DataLoader
from torchvision import datasets
from torchvision import  transforms
from lenet5 import Lenet5
from torch import nn
from torch import optim


def main():
    batchsz=32
    cifar_train=datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)#这是对加载一张图片进行设置,这个第一个参数为cifar是路径，在当前目录下下载
    cifar_train=DataLoader(cifar_train,batch_size=batchsz,shuffle=True)#这是对加载一批数据进行设置

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x,label=iter(cifar_train).next()#取到这个cifar_train的迭代器，用next()得到一个batch
    print('x:',x.shape,'label:',label.shape)

    #device=torch.device('cuda')
    #model=Lenet5().to(device)#实例化这个类，并搬到device上面
    #criteon=nn.CrossEntropyLoss().to(device)
    model = Lenet5()
    criteon=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    print(model)

    for epoch in range(2):#epoch大了之后就放到cpu上面跑

        model.train()#train模式
        for batchidx,(x,label) in enumerate(cifar_train):
            #x,label=x.to(device),label.to(device) model和数据要搬到cuda上面
            logits=model(x)
            loss=criteon(logits,label)

            #backprop
            optimizer.zero_grad()
            loss.backward()#向后传播计算导数
            optimizer.step()

        print(epoch,loss.item())#item()就是转换成numpy打印出来

        #test
        model.eval()#下面两行代码搭配使用，因为在test里面是不需要计算梯度的，这样安全一点
        with torch.no_grad():
            total_correct=0#记录一下预测正确的个数
            total_num=0#记录一下总共有多少张照片
            for x,label in cifar_test:
                #[b,10]
                logits=model(x)
                #[b]
                pred=logits.argmax(dim=1)#每张照片有10维的一个向量嘛，每个元记录了类别的可能性，把每张照片可能性最大的那个类别的所应找出来和label比较
                #pred:[b],label:[b]
                #两个向量eq一下，相同为1，不同为0，再将它们float一下加起来，就是预测正确图片的数量
                total_correct+=torch.eq(pred,label).float().sum().item()
                total_num+=x.size(0)#取x第一维的值，就是每次循环照片的数量
            #test数据预测完后，计算一下精确度
            acc=total_correct/total_num
            print(epoch,acc)


if __name__ == '__main__':
    main()