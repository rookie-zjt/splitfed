import socket
import time

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

def get_connect(address, max_retries=5):
    retries = 0
    while retries < max_retries:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            conn.connect(address)
            return conn
        except Exception as e:
            print(f"Failed {retries}/{max_retries}: {e}")
            conn.close()
            # 指数退避等待
            delay = 2 ** retries
            time.sleep(delay)
        finally:
            retries += 1
    raise Exception("Reached the maximum number of retry attempts.")

