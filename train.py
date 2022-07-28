import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from dataset.dataset import *
from utility.utils import *
from model import *


def train(net: VIN, trainloader, config, criterion, optimizer):
    print_header()
    # Automatically select device to make the code device agnostic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 做多个epoch，每个epoch
    for epoch in range(config.epochs):  # Loop over dataset multiple times
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, S1, S2, labels = [d.to(device) for d in data]
            if X.size()[0] != config.batch_size:
                continue  # Drop those data, if not enough for a batch
            net = net.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs, predictions = net(X, S1, S2, config.k)
            # Loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch, error_batch = get_stats(loss, predictions, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            num_batches += 1
        time_duration = time.time() - start_time
        # Print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)
    print('\nFinished training. \n')


def test(net: VIN, testloader, config):
    total, correct = 0.0, 0.0
    # Automatically select device, device agnostic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, labels = [d.to(device) for d in data]
        if X.size()[0] != config.batch_size:
            continue  # Drop those data, if not enough for a batch
        net = net.to(device)
        # Forward pass
        outputs, predictions = net(X, S1, S2, config.k)
        # Select actions with max scores(logits)
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data
        # Compute test accuracy
        correct += (torch.eq(torch.squeeze(predicted), labels)).sum()
        total += labels.size()[0]
    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))


if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    # 在terminal中命令行输入的时候，可以用 --xxx 指定的参数，假如不指定，运行采用默认值
    parser.add_argument(
        '--datafile',
        type=str,
        default='dataset/gridworld_8x8.npz',
        help='Path to data file')
    parser.add_argument('--imsize', type=int, default=8, help='Size of image')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.005,
        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument(
        '--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument(
        '--k', type=int, default=10, help='Number of Value Iterations')
    parser.add_argument(
        '--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument(
        '--l_h',
        type=int,
        default=150,
        help='Number of channels in first hidden layer')
    parser.add_argument(
        '--l_q',
        type=int,
        default=10,
        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    # 在terminal中命令行输入的这些参数，就形成了config
    config = parser.parse_args()
    # Get path to save trained model
    save_path = "trained/vin_{0}x{0}.pth".format(config.imsize)
    # Define Dataset
    # Dataset transformer: torchvision.transforms
    train_set = GridworldData(
        config.datafile, imsize=config.imsize, train=True, transform=None)
    # 测试的时候设置 train=False
    test_set = GridworldData(
        config.datafile,
        imsize=config.imsize,
        train=False,
        transform=None)
    # Create Dataloader
    # 训练模型的时候 shuffle=True，测试的时候 shuffle=False
    # batch_size表示有5个trainset,这5个是一样的吗？
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
    # Instantiate a VIN model
    net = VIN(config)
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=config.lr, eps=1e-6)
    # 因为本例中没有dropout，normalization等模块，所以没有添加
    # torch.train()和torch.eval()两个函数，也是没有问题的
    # Train the model
    train(net, train_loader, config, criterion, optimizer)
    # Test accuracy
    test(net, test_loader, config)
    # Save the trained model parameters
    torch.save(net.state_dict(), save_path)
