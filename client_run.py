import copy
import json
import os.path
import pickle
import random
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from model.client_side import ResNet18_client_side
from preparation.data_load import get_dataset
from preparation.data_preprocess import SkinData, train_transforms, test_transforms, dataset_iid, split
from util.tcp import *

# 随机数相关，保证结果相同
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))
# 有gpu就用cuda训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 客户端id
idx = config.this_cid

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs_train=None,
                 idxs_test=None):
        # self.net = net_client_model
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        # self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs_train), batch_size=config.batch_size, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=config.batch_size, shuffle=True)

        # 监听来自服务器的开始信号
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(config.clients_addr[idx])
        self.sock.listen(5)
        # 服务器与客户端的通信
        self.conn = None

    def __del__(self):
        self.close()

    def load_trainset(self):
        """
        Client loads the training dataset.
        """
        pass

    def close(self):
        self.sock.close()

    def accept_server_connect(self):
        print("waiting for server connecting...")
        conn,addr = self.sock.accept()
        self.conn = conn
        print("got conn!")

    def disconnect_server(self):
        self.conn.close()
        print("closed conn")


    def send_to_server(self, fx, y, iter, local_ep, idx, len_batch):
        """
        Sending activations(fx), labels(y), local_iter(iter), client_id(idx), len_batch to server
        """
        data = {
            "fx": fx,
            "y": y,
            "iter": iter,
            "l_ep": local_ep,
            "idx": idx,
            "len_b": len_batch
        }
        # 序列化
        serialized_data = pickle.dumps(data)
        send_data(serialized_data, self.conn)


    def send_to_fed(self, weights):
        """
        Sending weights to fed-server
        """
        # 序列化
        serialized_data = pickle.dumps(weights)
        send_data(serialized_data, self.conn)

    def recv_from_server(self):
        """
        Receiving gradients from MainServer
        """
        received_data = recv_data(self.conn)
        # 反序列化
        grad = pickle.loads(received_data)
        return grad


    def recv_from_fed(self):
        """
        Receiving global weights from FedServer
        """
        received_data = recv_data(self.conn)
        # 反序列化
        glob_weights = pickle.loads(received_data)
        return glob_weights

    def send_time_to_fed(self, ftimes, btimes):
        # 传递的是每个轮次所有批次训练时间列表
        data = {
            "f": ftimes,
            "b": btimes
        }
        serialized_data = pickle.dumps(data)
        send_data(serialized_data, self.conn)

    def train_client(self, net):
        """
        Client trains the model(front-part) on local dataset
        """
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
        # 记录训练时间
        forward_collect = []
        backward_collect = []
        # 可能本地迭代多轮，再给服务器发送激活向量(默认一轮)
        for local_i in range(self.local_ep):
            len_batch = len(self.ldr_train)
            # 每一轮有多个批次
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(f"client start to train batch {batch_idx}")
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                # ---------forward prop-------------
                forward_s = time.time()
                fx = net(images)
                # 记录前向传播时间
                forward_collect.append(time.time() - forward_s)
                client_fx = fx.clone().detach().requires_grad_(True)
                # print("to send ------")
                # 通信0: Sending activations to server
                self.send_to_server(client_fx, labels, local_i, self.local_ep, self.idx, len_batch)
                # print("sent ------")
                # --------backward prop -------------
                # 通信1: Receiving gradients from server
                dfx = self.recv_from_server()
                backward_s = time.time()
                fx.backward(dfx)
                # 记录后向传播时间
                backward_collect.append(time.time() - backward_s)
                optimizer_client.step()
                # print(f"client {idx} completed train {batch_idx}")
        # 发送本轮次的训练时间数据
        self.send_time_to_fed(forward_collect, backward_collect)
        return net.state_dict()

    def evaluate_client(self, net, ell):
        """
        Client test the model(front-part) on local dataset
        """
        net.eval()
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(images)
                # iter = ell
                # 通信 Sending activations to server
                self.send_to_server(fx, labels, ell, self.local_ep, self.idx, len_batch)
                # print(f"test batch {batch_idx} done!")

        return


#=========================================================

net_glob_client = ResNet18_client_side()


# 创建了一个ResNet18_client_side的实例，命名为net_glob_client
if torch.cuda.device_count() > 1:
    # 并判断是否有多个GPU可用，如果有则使用nn.DataParallel将模型分布到多个GPU上
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)

net_glob_client.to(device)
print(net_glob_client)
#==============================================


# 划分训练和测试集并增强数据
train, test = split(get_dataset())
dataset_train = SkinData(train, transform = train_transforms)
dataset_test = SkinData(test, transform = test_transforms)
# 划分不同客户端的数据集索引集合
idxs_train = dataset_iid(dataset_train, config.num_users)
idxs_test = dataset_iid(dataset_test, config.num_users)
#=========================================================
#创建一个Client对象，用于联邦学习的本地模型。
local = Client(net_glob_client, idx, config.lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs_train = idxs_train[idx], idxs_test = idxs_test[idx])

# 客户端模型的参数字典
w_glob_client = net_glob_client.state_dict()

# Training ------------------
# this epoch is global epoch, also known as rounds
for iter in range(config.epochs):
    # 每轮由服务器决定何时开始
    local.accept_server_connect()
    # print(f"epoch {iter} start")
    # Training -------------------
    w_client = local.train_client(net = copy.deepcopy(net_glob_client).to(device))
    # 通信2: send weights to fed-server
    local.send_to_fed(w_client)

    # Testing -------------------
    #评估本地模型在测试集上的性能，并打印出结果。
    local.evaluate_client(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
    local.disconnect_server()

    # 服务器与所有客户端都交互之后，再接收全局权重
    local.accept_server_connect()
    # print(f"get global weights")

    # Loading aggregated weights-------------------
    # 通信3: received global weights from fed-server
    w_glob_client = local.recv_from_fed()
    # Update client-side global model
    # 更新全局模型的权重为w_glob_client
    net_glob_client.load_state_dict(w_glob_client)
    local.disconnect_server()

#===================================================================================
local.close()
print("Training and Evaluation completed!")
