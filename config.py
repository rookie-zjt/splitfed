# 随机种子
SEED = 1234
# 参与的客户端数量
num_users = 1

batch_size = 256

epochs = 1

# participation of clients; if 1 then 100% clients participate in SFLV1
frac = 1

# learning rate
lr = 0.0001

model_name = 'resnet18'
# 模型分割位置（客户端最后一层）
# 取值范围 1-6（具体参考struct-split.png）
spilt = 6

dataset = 'HAM10000'
# 结果记录存放路径
save_path = './save'
# 数据集存放路径
data_path = './data'
# 当前客户端机器的id
this_cid = 0

clients_addr = [
    ('localhost', 12345),
    ('localhost', 23456),

    # ('192.168.99.11',12345),
    # ('192.168.99.113',23456),
]
# 可以不配置服务器地址
server_addr = ('localhost',22222)

# server_addr = ('192.168.99.1',11111)
