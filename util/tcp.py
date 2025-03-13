import json
import socket
import threading
import time

from util.file import send_file,receive_file

# 保证传入的序列化数据是字节序列(bytes)
def send_data(serialized_data, dest_address):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect(dest_address)
            client_socket.sendall(len(serialized_data).to_bytes(4, byteorder='big'))  # 大端表示的4字节数据长度
            client_socket.sendall(serialized_data)

            # src_address = client_socket.getsockname()
            # print(f"Data from {src_address} sent to {dest_address}")
        finally:
            client_socket.close()


def recv_data(conn_socket, src_address):
    serialized_data = bytearray()
    try:
        # print(f"Connected to {src_address}")
        # 接收数据长度
        data_length = conn_socket.recv(4)  # 接收4字节长度前缀
        data_length = int.from_bytes(data_length, byteorder='big')
        # 接收数据
        while len(serialized_data) < data_length:
            packet = conn_socket.recv(data_length - len(serialized_data))
            serialized_data.extend(packet)
        # print(f"Received {data_length}Byte data from {src_address}")
    finally:
        conn_socket.close()
        return serialized_data


def receive_multithreading_results(server_socket, server_port=11111):
    server_socket.bind(('localhost', server_port))
    server_socket.listen()
    print(f"Server is waiting for results on port {server_port}")
    while True:
        client_conn, client_address = server_socket.accept()
        # 为每个客户端连接创建一个新的线程
        client_thread = threading.Thread(target=recv_data, args=(client_conn, client_address))
        client_thread.start()


def receive_single_results(bind_addr=('localhost',11111)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(bind_addr)
        server_socket.listen(5)
        print(f" waiting for results......")
        client_conn, client_address = server_socket.accept()
        data = recv_data(client_conn, client_address)
        return data


def send_multi_file(file_path_list, dest_addr=('localhost',22222)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(dest_addr)
        for path in file_path_list:
            send_file(s, path)
            # 等待接收完成
            s.recv(4)
        # print(f"--- send {len(file_path_list)} files ---")


def receive_multi_file(recv_dir='.',cnt = 1, bind_addr=('localhost',11111)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(bind_addr)
        s.listen(5)
        conn, client_address = s.accept()
        with conn:
            for _ in range(cnt):
                receive_file(conn,recv_dir)
                # 接收完成
                conn.sendall(int(200).to_bytes(4, byteorder='big'))
            # print(f"--- received {cnt} files in {recv_dir} ---")


def send_msg(sock,msg_code=200):
    sock.sendall(msg_code.to_bytes(4,'big'))
    pass

def recv_msg(sock):
    pass