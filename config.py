SEED = 1234

num_users = 2

batch_size = 256

epochs = 2

frac = 1    # participation of clients; if 1 then 100% clients participate in SFLV1

lr = 0.0001 # learning rate

model_name = 'resnet18'

dataset = 'HAM10000'

save_path = './save'

send_path = './send'
recv_path = './recv'

data_path = './data'

# 当前服务器机器的id
this_cid = 0

clients_addr = [
    # ('localhost', 12345),
    # ('localhost', 23456),

    # ('0.0.0.0', 12345),
    # ('0.0.0.0', 23456),

    ('192.168.114.11',12345),
    ('192.168.114.113',23456),
]

# server_addr = ('localhost',22222)
# server_addr = ('0.0.0.0',22222)
server_addr = ('192.168.114.1',11111)
