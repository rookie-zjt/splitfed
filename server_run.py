import copy
import json
import pickle
import random
import time
import socket

import numpy as np
import torch
from torch import nn

import config
from config import *
from model.server_side import ResNet18_server_side, Baseblock
from util.tcp import *
from util.utility import *

# 随机数相关，保证结果相同
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))
# 有gpu就用cuda训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 通信相关函数==================================================
def server_send_to_client(dfx,conn):
    """
    Sending gradients to clients
    """
    # 序列化
    serialized_data = pickle.dumps(dfx)
    send_data(serialized_data, conn)

def fed_send_to_client(glob_weights,conn):
    """
    Sending global gradients to clients
    """
    # 序列化
    serialized_data = pickle.dumps(glob_weights)
    send_data(serialized_data, conn)

def server_recv_from_client(conn):
    """
    Receiving fx,y,and other from client
    """
    received_data = recv_data(conn)
    # 反序列化
    data = pickle.loads(received_data)
    return data['fx'], data['y'], data['iter'], data['l_ep'], data['idx'], data['len_b']

def fed_recv_from_client(conn):
    """
    Receiving local weights from client
    """
    received_data = recv_data(conn)
    # 反序列化
    local_w = pickle.loads(received_data)
    return local_w

def recv_time_from_client(conn):
    received_data = recv_data(conn)
    # 反序列化
    data = pickle.loads(received_data)
    # 返回的是两个时间列表
    return data['f'], data['b']

# 模型计算相关函数====================================================================
def FedAvg(w):
#FedAvg 函数负责对本地模型进行平均
#它接收一个本地模型的列表 (w)，其中每个元素都是一个包含模型权重的字典，然后返回一个字典 (w_avg)
    w_avg = copy.deepcopy(w[0])
    #将 w_avg 初始化为列表中第一个本地模型的深拷贝，也就是很多个本地模型，他们都是同样的格式，将第一个本地模型拷贝下来包括其模型格式
    for k in w_avg.keys():
        # 然后遍历字典中的每个键，每个键对于的就是那一层神经网络的权重的Tensor
        for i in range(1, len(w)):
            #从1开始遍历本地模型（因为第0层已经通过深拷贝存在w_avg了）
            w_avg[k] += w[i][k]
            #将所有的本地模型参数全部相加
        w_avg[k] = torch.div(w_avg[k], len(w))
    # w_avg 是一个字典，它存储了全局模型的参数。
    # k 是一个字符串，它表示参数的名称，例如 “layer1.0.weight” 或 “layer1.1.weight”。
    # w 是一个列表，它存储了每个客户端模型的参数字典。
    # torch.div 是一个 PyTorch 中的函数，它可以对两个张量（tensor）进行除法运算，并返回一个新的张量。
    # len 是一个 Python 中的函数，它可以返回一个列表的长度，即元素的个数，这里就是客户端的数目
    # 代码的意思是，对于每个参数 k，把 w_avg[k] 这个张量除以 w 这个列表的长度，也就是客户端的个数，然后把结果赋值给 w_avg[k]。这样就可以得到所有客户端模型参数的平均值。
    return w_avg

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float()/preds.shape[0]
    return acc

def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    #fx_client: 客户端设备上的特征向量
    #y: 客户端设备上的标签向量
    #idx: 服务器的索引
    #len_batch: 批次的个数
    # l_epoch_count: 客户端本地已经训练次数
    # l_epoch: 客户端本地需要训练次数
    # ps: l_epoch和epoch不是一个东西，l_epoch是指每一轮客户端可能本地多次
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr
    # 记录训练时间
    global server_forward_time_collect, server_backward_time_collect

    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr)

    # train and update
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    # ---------forward prop-------------
    forward_s = time.time()
    fx_server = net_server(fx_client)
    # 记录前向时间
    server_forward_time_collect.append(time.time()-forward_s)
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------
    backward_s = time.time()
    loss.backward()
    # 记录后向时间
    server_backward_time_collect.append(time.time()-backward_s)
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)

    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))

        # copy the last trained model in the batch
        w_server = net_server.state_dict()

        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch - 1:

            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not
            # We store the state of the net_glob_server()
            w_locals_server.append(copy.deepcopy(w_server))

            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)

            # print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            # collect the id of each new user
            if idx not in idx_collect:
                idx_collect.append(idx)
                # print(idx_collect)

        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display

            w_glob_server = FedAvg(w_locals_server)

            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)
            net_model_server = [net_glob_server for i in range(num_users)]

            w_locals_server = []
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    # send gradients to the client
    return dfx_client

def evaluate_server(fx_client, y, idx, len_batch, ell):
    #fx_client: 客户端设备上的特征向量
    #y: 客户端设备上的标签向量
    #idx: 服务器的索引
    #len_batch: 批次的个数
    #ell: 一个本地训练周期的长度
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server
    # net_model_server: 一个列表，存储了每个服务器上的模型
    # criterion: 一个损失函数，用来计算模型的损失
    # batch_acc_test: 一个列表，存储了每个批次的准确率
    # batch_loss_test: 一个列表，存储了每个批次的损失
    # check_fed: 一个布尔值，表示是否进行了联合操作
    # net_server: 一个模型，表示联合后的全局模型
    # net_glob_server: 一个模型，表示联合前的全局模型

    global loss_test_collect, acc_test_collect, count2, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    # loss_test_collect: 一个列表，存储了每次联合后的全局损失
    # acc_test_collect: 一个列表，存储了每次联合后的全局准确率
    # count2: 一个计数器，用来记录已经处理了多少个批次
    # num_users: 一个整数，表示参与联合学习的用户数量
    # acc_avg_train_all: 一个浮点数，表示所有用户在本地训练后的平均准确率
    # loss_avg_train_all: 一个浮点数，表示所有用户在本地训练后的平均损失
    # w_glob_server: 一个字典，存储了每个用户在本地训练后的模型权重
    # l_epoch_check: 一个布尔值，表示是否完成了一个本地训练周期
    # fed_check: 一个布尔值，表示是否进行了联合操作

    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    # loss_test_collect_user: 一个列表，存储了每次训练后的用户损失
    # acc_test_collect_user: 一个列表，存储了每次训练后的用户准确率

    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()
    #复制net_model_server[idx]到net，并将其设置为评估模式，因为不训练仅仅用于评估

    with torch.no_grad():
        #不计算梯度，将fx_client和y转移到设备上
        fx_client = fx_client.to(device)
        y = y.to(device)
        #---------forward prop-------------
        fx_server = net(fx_client)
        #使用net对fx_client进行预测，并得到fx_server

        # calculate loss
        loss = criterion(fx_server, y)
        #使用criterion计算fx_server和y之间的损失
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        #使用calculate_accuracy函数计算准确率


        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        #将损失和准确率添加到batch_loss_test和batch_acc_test列表中


        count2 += 1
        #增加count2
        if count2 == len_batch:
            #如果count2等于len_batch，说明已经处理了一个批次，则计算这个批次的平均损失和平均准确率
            #并清空batch_loss_test和batch_acc_test列表，并重置count2。
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))

            # if a local epoch is completed
            if l_epoch_check:
                #如果l_epoch_check为真，说明已经完成了一个本地训练周期
                #则将这个周期最后一次评估得到的平均损失和平均准确率存储到loss_test_collect_user和acc_test_collect_user列表中，并将l_epoch_check设置为假。
                l_epoch_check = False

                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            # if federation is happened----------
            if fed_check:
                #如果fed_check为真，说明已经进行了联合操作，则计算所有用户在本地训练后的平均损失和平均准确率
                #并将它们存储到loss_test_collect和acc_test_collect列表中，并清空loss_test_collect_user和acc_test_collect_user列表
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")

                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")

    return

#===================================================================

# 服务器端的全局模型
net_glob_server = ResNet18_server_side([2,2,2], 7) #7 is my numbr of classes

# 基于 ResNet18 的卷积神经网络，使用 Baseblock 作为基本模块，有三个卷积层，[2,2,2]，每层有 2 个 Baseblock，最后有 7 个类别的输出。
if torch.cuda.device_count() > 1:
    # 是否有多个 GPU 设备
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs

net_glob_server.to(device)

print(net_glob_server)

# 训练过程中的相关数据记录===================================================================================
# 每个epoch的损失和准确率
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
# 每个batch的损失和准确率
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 训练和测试的批次计数
count1 = 0
count2 = 0

# For Client Side Loss and Accuracy
# 各个客户端的损失和准确率
loss_avg_all_user_train = 0
acc_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

# 用于存储每个客户端模型的参数字典
w_locals_server = []

# Initialization of net_model_server and net_server (server-side model)
# 服务端模型
net_server = copy.deepcopy(net_glob_server).to(device)
# 全局模型列表，每个客户端都拥有一个独立的服务端模型。
net_model_server = [net_glob_server for i in range(num_users)]

#client idx collector
# 用于存储每轮选择的客户端的索引，每轮都会随机选择一部分客户端来参与（默认frac=1则全部参与）。
idx_collect = []

# 记录时间----------------------------------
# 轮次时间
epoch_time_collect = []
# 下面几个列表，都是以批次为单位记录
# 服务端前向传播
server_forward_time_collect = []
# 服务端后向传播
server_backward_time_collect = []
# 客户端前向传播
clients_forward_time_collect = [[]for i in range(num_users)]
# 客户端后向传播
clients_backward_time_collect = [[]for i in range(num_users)]
# 通信时间(四种通信方式）
# 分别对应client -> server/ server -> client/ client -> fed/ fed -> client)
communication_time_collect = [[[]for j in range(4)]for i in range(num_users)]

# 是否需要在每个客户端上进行本地训练
# 如果为 True，就表示需要在每个客户端上训练一定轮数（epochs）的数据，然后把模型的参数发送给服务器；
# 如果为 False，就表示不需要在每个客户端上进行本地训练，直接使用服务器上的全局模型。
l_epoch_check = False
# 是否需要在服务器上进行联邦平均算法
# 如果为 True，就表示需要在服务器上计算所有客户端模型的参数的平均值，并把全局模型的参数发送给所有客户端；
# 如果为 False，就表示不需要在服务器上进行联邦平均算法，直接使用每个客户端自己的模型。
fed_check = False

#============================================================

# 服务器模型的参数字典
w_glob_server = net_glob_server.state_dict()


program = f"SFLV1_split{spilt} ResNet18 on HAM10000 "
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files
start = time.time()
for iter in range(epochs):
    epoch_start = time.time()
    # 每一轮训练，随机选择一部分用户（默认全部选择）
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    # 各个本地客户端的参数字典
    w_locals_client = []

    for idx in idxs_users:
        # print(f"start client {idx}")
        # 由服务器发起连接
        conn = get_connect(config.clients_addr[idx])

        # Training -----------------
        while True:
            batch_start = time.time()
            # print(f"server start to train batch {count1}")
            # main-server------
            # 通信0: receive data from client
            server_recv_s = time.time()
            fx, y, local_iter, local_ep, _, len_batch = server_recv_from_client(conn)
            # 记录接收客户端i的时间
            communication_time_collect[idx][0].append(time.time() - server_recv_s)
            # print("received ------")
            # 服务端的前向和后向传播
            dfx = train_server(fx, y, local_iter, local_ep, idx, len_batch)

            # 通信1: send gradients to client
            server_send_s = time.time()
            server_send_to_client(dfx, conn)
            # 记录发送给客户端i的时间
            communication_time_collect[idx][1].append(time.time() - server_send_s)

            print(f"client {idx} completed train {count1} in {(time.time()-batch_start)/60} min")
            # 计数器清零说明所有批次完成（train_server中每个批次计数）
            if count1 == 0:
                break

        # 接收客户端训练时间数据(列表)
        f_times, b_times = recv_time_from_client(conn)

        # 记录客户端i前向传播的时间
        clients_forward_time_collect[idx].extend(f_times)
        # 记录客户端i后向传播的时间
        clients_backward_time_collect[idx].extend(b_times)

        # fed-server------
        # 通信2: receive weights from client
        fed_recv_s = time.time()
        w_client = fed_recv_from_client(conn)
        # 记录联邦接收客户端i的时间
        communication_time_collect[idx][2].append(time.time() - fed_recv_s)
        # 并将本地模型的权重w_client添加到w_locals_client列表中
        w_locals_client.append(copy.deepcopy(w_client))

        # Testing -------------------
        while True:
            batch_start = time.time()
            # print(f"test batch {count2} start!")
            # 通信 receive data from client
            fx_, y_, _, local_ep_, _, len_batch_ = server_recv_from_client(conn)
            evaluate_server(fx_,y_,idx,len_batch_,iter)
            # print(f"client {idx} completed test {count2} in {(time.time()-batch_start)/60} min")
            # 计数器清零说明所有批次完成（evaluate_server中每个批次计数）
            if count2 == 0:
                break
        conn.close()

    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")
    #使用FedAvg函数来计算w_locals_client列表中所有本地模型权重的平均值，并将其赋值给w_glob_client。
    w_glob_client = FedAvg(w_locals_client)

    # Update client-side global model
    for idx in idxs_users:
        # 每一轮由服务器发起连接，决定哪个客户端先开始
        conn = get_connect(config.clients_addr[idx])
        # 通信3: send global weights to all client
        fed_send_s = time.time()
        fed_send_to_client(w_glob_client, conn)
        # 记录联邦分发给客户端i的时间
        communication_time_collect[idx][3].append(time.time() - fed_send_s)
        conn.close()

    epoch_time_collect.append(time.time()-epoch_start)
    print(f"epoch time: {epoch_time_collect[-1]} second")

#===================================================================================

print("Training and Evaluation completed!")
print(f"total time: {time.time()-start} second")

# 训练结束，记录相关数据
save_result(program, acc_train_collect, acc_test_collect, loss_train_collect, loss_test_collect, epoch_time_collect, config.save_path)

record_time(program, server_forward_time_collect, server_backward_time_collect,
            clients_forward_time_collect, clients_backward_time_collect, communication_time_collect, config.save_path)


