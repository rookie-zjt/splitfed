import os

from pandas import DataFrame


def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


# Save output data to .excel file (we use for comparision plots)
def save_result(program_name, acc_train_collect, acc_test_collect, epoch_time_collect,save_dir ='.'):
    # 联邦学习的轮数
    round_process = [i for i in range(1, len(acc_train_collect) + 1)]
    # df三个数据列分别对应轮数、训练准确率和测试准确率。
    df = DataFrame({
        'round': round_process,
        'acc_train': acc_train_collect,
        'acc_test': acc_test_collect,
        'epoch_time': epoch_time_collect
    })
    file_name = os.path.join(save_dir, program_name + ".xlsx")
    # index参数为False，表示不保存行索引
    df.to_excel(file_name, sheet_name="v1_test", index=False)


