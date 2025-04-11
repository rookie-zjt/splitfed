# splitfed

具体算法代码说明可以参考note.md

## 数据集 HAM10000
kaggle下载
```
#!/bin/bash
curl -L -o ~/Downloads/skin-cancer-mnist-ham10000.zip\
  https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000
```

## 必要运行环境
推荐使用conda或者miniforge管理

    python == 3.6/3.7
    torch == 1.8.0/1.12.1
    torchvision == 0.9.0/0.13.1 (对应torch版本)
    pandas    
    scikit-learn
    pillow
    numpy

## 运行说明

运行前需要配置config.py的相关参数

    num_users：整数，参与的客户端数量
    this_cid：整数，每个客户端都不同，范围[0,num_users)
    clients_addr：列表，长度不小于num_users，每个元素对应客户端地址(ip,port)


客户端运行
`python client_run.py`

服务器运行
`python server_run.py`

