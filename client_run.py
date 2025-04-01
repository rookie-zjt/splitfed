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
from preparation.data_load import get_dataset, split
from preparation.data_preprocess import SkinData, train_transforms, test_transforms, dataset_iid
from util import tcp

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
        Sending activations(fx), labels(y), client_id(idx), len_batch to server
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
        tcp.send_data(serialized_data, self.conn)
        '''
       # 文件形式保存
        flist = [f'{config.send_path}/other{idx}.json',f'{config.send_path}/y{idx}.pth',f'{config.send_path}/fx{idx}.pth']
        dict = {
            "iter": iter,
            "l_ep": local_ep,
            "idx": idx,
            "len_b": len_batch
        }
        js_data = json.dumps(dict)
        with open(flist[0],'w',encoding='utf-8') as f:
            f.write(js_data)
        torch.save(y, flist[1])
        torch.save(fx, flist[2])
        # 传多个文件
        tcp.send_multi_file(flist, config.server_addr)
    '''

    def send_to_fed(self, weights):
        """
        Sending weights to fed-server
        """
        # 序列化
        serialized_data = pickle.dumps(weights)
        tcp.send_data(serialized_data, self.conn)
        '''
        torch.save(weights, f'{config.send_path}/weights{idx}.pth')
        tcp.send_multi_file([f'{config.send_path}/weights{idx}.pth'], config.server_addr)
        '''
    def send_time_to_fed(self, ftimes, btimes):
        # 传递的是每个轮次所有批次训练时间列表
        data = {
            "f": ftimes,
            "b": btimes
        }
        serialized_data = pickle.dumps(data)
        tcp.send_data(serialized_data, self.conn)

    def recv_from_server(self):
        """
        Receiving gradients from MainServer
        """
        received_data = tcp.recv_data(self.conn)
        # 反序列化
        grad = pickle.loads(received_data)
        '''
        # 接收文件
        tcp.receive_multi_file(config.recv_path,1,config.clients_addr[0])
        # 加载文件
        data = torch.load(f'{config.recv_path}/grad{idx}.pth')
        '''
        return grad


    def recv_from_fed(self):
        """
        Receiving global weights from FedServer
        """
        received_data = tcp.recv_data(self.conn)
        # 反序列化
        glob_weights = pickle.loads(received_data)
        '''
        # 接收文件
        tcp.receive_multi_file(config.recv_path,1,config.clients_addr[0])
        # 加载文件
        data = torch.load(f'{config.recv_path}/glob_grad{idx}.pth')
        '''
        return glob_weights

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
                print(f"server start to train batch {batch_idx}")
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                # ---------forward prop-------------
                forward_s = time.time()
                fx = net(images)
                # 记录前向传播时间
                forward_collect.append(time.time() - forward_s)
                client_fx = fx.clone().detach().requires_grad_(True)
                print("to send ------")
                # 通信 Sending activations to server
                # 顺便传递前向传播时间
                # dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                self.send_to_server(client_fx, labels, local_i, self.local_ep, self.idx, len_batch)
                print("received ------")
                # --------backward prop -------------
                # 通信 Receiving gradients from server
                dfx = self.recv_from_server()
                backward_s = time.time()
                fx.backward(dfx)
                # 记录后向传播时间
                backward_collect.append(time.time() - backward_s)
                optimizer_client.step()
                # print(f"train batch {batch_idx} done!")
                print(f"client {idx} completed train {batch_idx}")
        # 发送本轮次的训练时间数据
        self.send_time_to_fed(forward_collect,backward_collect)
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

                # evaluate_server(fx, labels, self.idx, len_batch, ell)
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


# 划分数据集并增强数据
train, test = split(get_dataset())
dataset_train = SkinData(train, transform = train_transforms)
dataset_test = SkinData(test, transform = test_transforms)
# 划分不同客户端的数据集索引集合
idxs_train = dataset_iid(dataset_train, config.num_users)
idxs_test = dataset_iid(dataset_test, config.num_users)
#=========================================================
#创建一个Client对象，用于联邦学习的本地模型。
local = Client(net_glob_client, idx, config.lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs_train = idxs_train[idx], idxs_test = idxs_test[idx])
#copy weights
w_glob_client = net_glob_client.state_dict()
# Training ------------------
# this epoch is global epoch, also known as rounds
for iter in range(config.epochs):
    # 每轮由服务器决定何时开始
    local.accept_server_connect()
    print(f"epoch {iter} start")
    # Training -------------------
    w_client = local.train_client(net = copy.deepcopy(net_glob_client).to(device))
    # 通信 send weights to fed-server
    local.send_to_fed(w_client)

    # Testing -------------------
    #评估本地模型在测试集上的性能，并打印出结果。
    local.evaluate_client(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
    local.disconnect_server()

    # 服务器与所有客户端都交互之后，再接收全局权重
    local.accept_server_connect()
    print(f"get global weights")
    # Loading aggregated weights-------------------
    # 通信 received global weights from fed-server
    w_glob_client = local.recv_from_fed()
    # Update client-side global model
    #使用net_glob_client.load_state_dict函数来更新全局模型的权重为w_glob_client
    net_glob_client.load_state_dict(w_glob_client)
    local.disconnect_server()

#===================================================================================
local.close()
print("Training and Evaluation completed!")