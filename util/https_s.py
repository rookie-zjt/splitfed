import requests
from flask import Flask, request, jsonify

import os

app = Flask(__name__)
SAVE_FOLDER = './gradients'
os.makedirs(SAVE_FOLDER, exist_ok=True)  # 创建保存梯度的目录



def upload_glob_w(id, file, dest_url):
    # --- 1. 计算分块参数 ---
    chunk_size = 1024 * 1024  # 每块1MB（可根据网络调整）
    total_size = os.path.getsize(file)  # 获取文件总大小
    total_chunks = (total_size + chunk_size - 1) // chunk_size  # 计算总分块数

    # --- 2. 分块读取并上传 ---
    with open(file, 'rb') as f:
        for i in range(total_chunks):
            # 读取当前分块数据
            chunk_data = f.read(chunk_size)

            # 构建请求头
            headers = {
                'Client-ID': id,
                'Chunk-Index': str(i),
                'Total-Chunks': str(total_chunks)
            }

            # --- 3. 发送分块数据 ---
            try:
                response = requests.post(
                    dest_url,
                    headers=headers,
                    data=chunk_data,  # 二进制数据直接发送
                    verify=False,
                    # verify='server.crt'  # 验证服务端证书
                )
                if response.status_code != 200:
                    print(f"分块 {i} 上传失败：{response.text}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"网络错误：{str(e)}")
                return False

            print(f"分块 {i}/{total_chunks} 上传成功")
    return True

@app.route('/local_weights', methods=['POST'])
def receive_weights():
    # --- 1. 验证请求头部信息 ---
    client_id = request.headers.get('Client-ID')  # 从请求头获取客户端ID
    chunk_index = request.headers.get('Chunk-Index')  # 当前分块的序号（从0开始）
    total_chunks = request.headers.get('Total-Chunks')  # 总分块数量

    # 校验必需的头信息是否存在
    if not all([client_id, chunk_index, total_chunks]):
        return jsonify({'error': '缺少必要头部信息'}), 400

    # --- 2. 保存分块数据 ---
    save_path = os.path.join(SAVE_FOLDER, client_id, f'{client_id}_weights.part{chunk_index}')
    with open(save_path, 'wb') as f:
        f.write(request.data)  # 将二进制数据写入文件

    # --- 3. 返回成功响应 ---
    return jsonify({'message': f'分块 {chunk_index}/{total_chunks} 上传成功'}),200


# 使用示例

if __name__ == '__main__':
    # 启动HTTPS服务，可使用SSL证书加密通信
    app.run(
        host='0.0.0.0',
        port=443,
        # ssl_context=('server.crt', 'server.key')  # 证书文件路径
    )

