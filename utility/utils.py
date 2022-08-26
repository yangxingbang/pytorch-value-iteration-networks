import numpy as np
import torch


def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float): rep = "%g" % x
    else: rep = str(x)
    return " " * (l - len(rep)) + rep


def get_stats(loss, predictions, labels):
    # 8个可能动作中只最大的那个作为结果动作，取它的索引
    cp = np.argmax(predictions.cpu().data.numpy(), 1)
    # 求 cp 和 labels 不相等 占 （相等+不相等） 的比率， 就能表示出整个序列中有多少比例的动作是不一致的
    # 这样评价误差能得到神经网络计算出的路径和最短路径的差异有多大
    error = np.mean(cp != labels.cpu().data.numpy())
    return loss.item(), error


def print_stats(epoch, avg_loss, avg_error, num_batches, time_duration):
    print(
        fmt_row(10, [
            epoch + 1, avg_loss / num_batches, avg_error / num_batches,
            time_duration
        ]))


def print_header():
    print(fmt_row(10, ["Epoch", "Train Loss", "Train Error", "Epoch Time"]))
