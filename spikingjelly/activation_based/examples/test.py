import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime

def main():
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps') #模拟的时间步数，默认为16。
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=8, type=int, help='batch size') #批次大小，默认为8。
    parser.add_argument('-epochs', default=64, type=int, metavar='N',help='number of total epochs to run') #要运行的总时代数，默认为64。
    parser.add_argument('-j', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)') #数据加载工作程序的数量，默认为4。
    parser.add_argument('-data-dir', type=str, help='root dir of DVS Gesture dataset') #DVS手势数据集的根目录。
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint') #用于保存日志和检查点的根目录。
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path') #从检查点路径恢复。
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training') #自动混合精度训练。
    parser.add_argument('-cupy', action='store_true', help='use cupy backend') #使用cupy后端。
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam') #使用哪种优化器。SDG或Adam。
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD') #SGD的动量，默认为0.9。
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate') #学习率，默认为0.1。
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN') #CSNN的通道数，默认为128。

    args = parser.parse_args() #解析参数。
    print(args)

    net = parametric_lif_net.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True) #创建模型。

    functional.set_step_mode(net, 'm') #设置模型的步骤模式为'm'。
    if args.cupy:   #如果使用cupy后端。
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)    #设置模型的后端为cupy。

    print(net)

    net.to(args.device) #将模型移动到设备上。

    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')   #创建测试集。


    test_data_loader = torch.utils.data.DataLoader( #创建测试数据加载器。
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    """
     for i,(x,target)in enumerate(test_data_loader):
        print(x.size())
        print(target.size())
    """

    #exit()
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')    #设置输出目录。

    if args.amp:    #如果使用自动混合精度训练。
        out_dir += '_amp'   #在输出目录中添加'_amp'。

    if args.cupy:   #如果使用cupy后端。
        out_dir += '_cupy'  #在输出目录中添加'_cupy'。

    if not os.path.exists(out_dir): #如果输出目录不存在。
        os.makedirs(out_dir)    #创建输出目录。
        print(f'Mkdir {out_dir}.')  #打印'Mkdir {out_dir}.'。

    writer = SummaryWriter(out_dir) #创建摘要写入器。

    if args.resume: #如果从检查点路径恢复。
        checkpoint = torch.load(args.resume, map_location='cpu')    #加载检查点。
        net.load_state_dict(checkpoint['net'])  #加载模型的状态字典。

    net.eval()  #设置模型为评估模式。
    test_loss = 0   #测试损失。
    test_acc = 0    #测试准确率。
    test_samples = 0    #测试样本数。
    with torch.no_grad():   #不进行梯度计算。
        for frame, label in test_data_loader:   #遍历测试数据加载器。
            frame = frame.to(args.device)   #将帧移动到设备上。
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W] #转置帧。
            label = label.to(args.device)   #将标签移动到设备上。
            label_onehot = F.one_hot(label, 11).float()   #将标签转换为one-hot编码。
            out_fr = net(frame).mean(0)   #前向传播。
            loss = F.mse_loss(out_fr, label_onehot)  #计算损失。
            test_samples += label.numel()   #测试样本数加上标签的元素数。
            test_loss += loss.item() * label.numel()        #测试损失加上损失乘以标签的元素数。
            test_acc += (out_fr.argmax(1) == label).float().sum().item()    #测试准确率加上预测结果的元素数。
            functional.reset_net(net)   #重置模型。
    test_loss /= test_samples   #测试损失除以测试样本数。
    test_acc /= test_samples    #测试准确率除以测试样本数。
    writer.add_scalar('test_loss', test_loss)   #写入测试损失。
    writer.add_scalar('test_acc', test_acc)    #写入测试准确率。

    print(args)
    print(out_dir)
    print(f'test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}')

if __name__ == '__main__':
    main()