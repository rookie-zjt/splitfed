import json
import socket
import threading
import time

from util.file import send_file,receive_file

# 保证传入的序列化数据是字节序列(bytes)
def send_data(serialized_data, conn_socket):
    msg = 0
    while msg != 200:
        # 8字节长度前缀
        conn_socket.sendall(len(serialized_data).to_bytes(8, byteorder='big'))
        conn_socket.sendall(serialized_data)
        # 等待接收完成
        msg = conn_socket.recv(2)
        msg = int.from_bytes(msg, byteorder='big')
    return msg

def recv_data(conn_socket):
    serialized_data = bytearray()
    # 接收数据长度
    data_length = conn_socket.recv(8)
    length = int.from_bytes(data_length, byteorder='big')
    # 接收数据
    while len(serialized_data) < length:
        packet = conn_socket.recv(length - len(serialized_data))
        serialized_data.extend(packet)
    # 接收成功
    conn_socket.sendall(int(200).to_bytes(2, byteorder='big'))
    return serialized_data


def send_multi_file(file_path_list, dest_addr=('localhost',22222)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.connect(dest_addr)
        for path in file_path_list:
            send_file(s, path)
            # 等待接收完成
            s.recv(4)
        # print(f"--- send {len(file_path_list)} files ---")


def receive_multi_file(recv_dir='.',cnt = 1, bind_addr=('localhost',11111)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(bind_addr)
        s.listen(5)
        conn, client_address = s.accept()
        with conn:
            for _ in range(cnt):
                receive_file(conn,recv_dir)
                # 接收完成
                conn.sendall(int(200).to_bytes(4, byteorder='big'))
            # print(f"--- received {cnt} files in {recv_dir} ---")
