import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        # 使用 detach() 计算中间结果
        intermediate_output = x.detach()
        x = torch.relu(x)
        x = self.fc2(x)
        return x, intermediate_output


# 创建网络和优化器
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建一个输入张量
input_tensor = torch.randn(1, 10, requires_grad=True)

# 前向传播
output, intermediate_output = model(input_tensor)

# 使用中间结果进行某些操作（例如，记录或分析）
print("Intermediate output (detached):", intermediate_output)

# 计算损失
target = torch.tensor([[0.0, 1.0]])
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

# 反向传播
loss.backward()

# 更新参数
optimizer.step()
