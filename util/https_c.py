import requests
import os

def upload_weights(client_id, gradient_file, server_url):
    # --- 1. 计算分块参数 ---
    chunk_size = 1024 * 1024  # 每块1MB（可根据网络调整）
    total_size = os.path.getsize(gradient_file)  # 获取文件总大小
    total_chunks = (total_size + chunk_size - 1) // chunk_size  # 计算总分块数

    # --- 2. 分块读取并上传 ---
    with open(gradient_file, 'rb') as f:
        for i in range(total_chunks):
            # 读取当前分块数据
            chunk_data = f.read(chunk_size)

            # 构建请求头
            headers = {
                'Client-ID': client_id,
                'Chunk-Index': str(i),
                'Total-Chunks': str(total_chunks)
            }

            # --- 3. 发送分块数据 ---
            try:
                response = requests.post(
                    server_url,
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


upload_weights(
    client_id='client_001',
    gradient_file='./weights.pth',
    server_url='http://127.0.0.1:443/upload_gradient'
)