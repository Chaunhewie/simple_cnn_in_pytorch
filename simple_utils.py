# coding=utf-8
'''
Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
'''

Total_Bar_Length = 30


def progress_bar(cur, tot, msg):
    '''
    打印训练进度
    '''
    s = "\r["
    prog = int(float(Total_Bar_Length) * (cur + 1) / tot)
    rest = Total_Bar_Length - prog - 1
    s = s + "=" * prog + ">" + "." * rest + "]"
    s += " | " + msg
    if cur < tot - 1:
        print(s, end="")
    else:
        print(s)

def num_of_parameters_of_net(net):
    '''
    打印网络的参数个数
    '''
    num_of_parameters = 0
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
        # print(parameters)
        num = 1
        for i in parameters.size():
            num *= i
        print(num)
        num_of_parameters += num
    return num_of_parameters


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
            
