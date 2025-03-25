## 联邦拆分学习 spiltfed learning SFL

数据集
HAM10000
```
#!/bin/bash
curl -L -o ~/Downloads/skin-cancer-mnist-ham10000.zip\
  https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000
```
参考代码
https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning

sflv1.py(含详细注释)


- 框架
![总体框架图](image.png)
    - fed server
    - client-side
    - main server

**每轮迭代流程**
- 每一批次batch都需要服务器和客户端交换数据，所有批次结束，使用了整个数据集所有样本一次
- 客户端可以本地迭代多轮local_epoch，默认本地迭代只有一轮
- 只有所有客户端本地迭代均完成，才算完成一次全局迭代epoch

*每个batch过程中*

client (前半部分forward) --(激活层输出，样本标签)--> main server（后半部分forward）\
client (前半部分back) <--(梯度)-- main server（后半部分back）

*每个local_epoch结束后*

client --(本地权重)--> fed 

*每个epoch结束后*

client <--(整体平均化权重)-- fed 


### 算法
- 服务端server
    - main 模型的后半部分的前向/后向传播
    1. 注意每次client传来的粉碎数据都要处理，并保存梯度副本用于最后更新
    2. 更新权重是在所有client的数据都处理过之后，即每轮最后

    - fed 聚合client的梯度，回传给client
    1. 不需要有模型，只需要记录各个client权重的副本，用于聚合
    2. 在各个client后向传播结束后进行聚合
    
- 客户端client
0. 各个client所用数据集应该互斥，模拟实际情况
1. 模型的前半部分的前向传播
2. 等main前后传播之后，再后向传播
3. 等fed进行聚合更新

#### 两个变体
- 基于服务端的聚合 sfl-v1

- 基于数据标签共享 sfl-v2

