import os
from pandas import DataFrame, concat

def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


# Save output data to .excel file (we use for comparision plots)
def save_result(program_name, acc_train_collect, acc_test_collect, loss_train_collect, loss_test_collect, epoch_time_collect, save_dir = "."):
    # 联邦学习的轮数
    round_process = [i for i in range(1, len(acc_train_collect) + 1)]

    df = DataFrame({
        'round': round_process,
        'acc_train': acc_train_collect,
        'acc_test': acc_test_collect,
        'loss_train': loss_train_collect,
        'loss_test': loss_test_collect,
        'epoch_time': epoch_time_collect
    })
    file_name = os.path.join(save_dir, program_name + ".xlsx")
    # index参数为False，表示不保存行索引
    df.to_excel(file_name, sheet_name="v1_test", index=False)

def record_time(program_name,
                server_f_collect,
                server_b_collect,
                clients_f_collect,
                clients_b_collect ,
                communication_collect,
                save_dir = "."):
    # 总共迭代次数
    iteration = [i for i in range(1, len(server_f_collect)+1) ]
    # 用户数
    users= len(clients_f_collect)
    df1 = DataFrame({
        "iter": iteration,
        "server_forward":server_f_collect,
        "server_backward": server_b_collect
    })
    # 多维list转dataframe
    # 多个客户端
    df2 = DataFrame(clients_f_collect, index=[f"client_{i}_forward" for i in range(len(clients_f_collect))]).T
    df3 = DataFrame(clients_b_collect, index=[f"client_{i}_backward" for i in range(len(clients_b_collect))]).T
    # 多个客户端，4个通信方式
    flat_head = [f"client_{i}_communicate_{j}"  for i in range(len(communication_collect)) for j in range(4)]
    flat_communication = [sublist for block in communication_collect for sublist in block]
    df4 = DataFrame(flat_communication, index=flat_head).T
    # print(df1.shape,df2.shape,df3.shape,df4.shape)
    # 按照列合并（列增多）
    df = concat([df1,df2,df3,df4],axis=1)


    file_name = os.path.join(save_dir, program_name + "time.xlsx")
    # index参数为False，表示不保存行索引
    df.to_excel(file_name, sheet_name="v1", index=False)



