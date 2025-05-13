### 参考教程
https://blog.csdn.net/friendshiptang/category_11985465.html
https://blog.csdn.net/qq_42078934/category_12246150.html
https://blog.csdn.net/huiyuanliyan/category_9291206.html

jetson环境可以配置参考以上三个博主的帖子

https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

英伟达nvidia论坛上的这个帖子，有pytorch相关问题的解答讨论（有相应版本的pip whl下载）

### jeston-stats(jtop)
可以实时查看相关状态
- 安装
sudo pip install jetson-stats
- 使用
sudo jtop


### 手动安装cuda和cudnn
- cuda需要10.2
`sudo apt-get install cuda-toolkit-10-2`
- cudnn需要8
`sudo apt-get install libcudnn8`

.bashrc文件配置
```
# cuda (nvcc)
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_ROOT=/usr/local/cuda

```


### Miniforge管理python环境（使用方法conda一致）
需要安装：
- python使用3.6版本
- numpy  不安装会导致torch导入失败
- pandas
- scikit-learn (sklearn)
- pillow (PIL)  torchvision编译需要


- torch 用pip通过whl轮子安装，直接用conda安装没尝试过（jetons4.6对应torch1.6-1.10）
  安装时，可能出现：
    - 安装numpy时可能缺少cython————手动安装cython
    - 超时————修改超时时间，添加参数--default-timeout=1000

- torchvision 编译源码安装，直接conda安装没尝试过 (torch1.8对应0.9.0)
  从github下载相应版本的源码(约1GB)
  github地址 https://github.com/pytorch/vision/
   - 进入vision-0.9.0目录 cd   vision-0.9.0
   - 指定版本 export BUILD_VERSION=0.9.0
   - 编译安装 python3 setup.py install --user

  编译安装过程可能较长


----
### 清华源配置
*清华源缺少一些包，不太建议*

pip

`pip config set global.extra-index-url "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"`

---
conda

1. 创建.condarc文件，内容如下：
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```
2. 清除索引
`conda clean -i`