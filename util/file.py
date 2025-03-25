import hashlib
import socket
import json
import os

def send_file(sock, file_path):
    # 发送文件名和文件大小
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    file_info = {'file_name': file_name, 'file_size': file_size}
    sock.sendall(json.dumps(file_info).encode("utf-8"))
    # print("send info:\n",file_info)
    # 等待接收完成
    sock.recv(4)
    # 发送文件内容
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(4096)
            if not data:
                break
            sock.sendall(data)


def receive_file(sock, recv_dir='.'):
    # 接收文件信息
    file_info_json = b''
    while True:
        data = sock.recv(1024)
        file_info_json += data
        if len(data) < 1024:
            break
    # print('recv info:\n',file_info_json)
    # 接收完成
    sock.sendall(int(200).to_bytes(4, byteorder='big'))
    file_info = json.loads(file_info_json.decode("utf-8"))

    # 接收文件内容
    file_name = os.path.join(recv_dir, file_info['file_name'])
    file_size = file_info['file_size']
    received_size = 0
    with open(file_name, 'wb') as file:
        while received_size < file_size:
            data = sock.recv(4096)
            file.write(data)
            received_size += len(data)

def file_hash(file_path):
    # 文件哈希
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def cmp_file(file1_path, file2_path):
    # 比较文件是否一致
    return file_hash(file1_path) == file_hash(file2_path)

# def combine_chunks(file_name, total_chunks):
#     with open(file_name, 'wb') as outfile:
#         for i in range(total_chunks):
#             chunk_path = f'{file_name}.part{i}'
#             with open(chunk_path, 'rb') as infile:
#                 outfile.write(infile.read())
#             os.remove(chunk_path)
