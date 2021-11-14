def adjust_learning_rate(optimizer, epoch, lr, epochs):
    """Sets the learning rate to the initial LR decayed by 10 every epochs / 3 epochs"""
    lr = lr * (0.1 ** (epoch // int(epochs/3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr