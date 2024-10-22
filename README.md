# 1. 创建conda虚拟环境
```
create -n demo python=3.12
conda activate demo
```

# 2. 如果你有显卡安装pytorch-gpu, 如果你没有显卡安装pytorch-cpu
## GPU版本
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
## CPU版本
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

# 3. pip安装对应的包
```
pip install -r requirements.txt
```

# 4. 下载数据集
```
https://www.kaggle.com/c/dogs-vs-cats/data
```

# 5. 修改simple_classification.ipynb数据集载入部分, 尝试运行代码

