import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# bitsandbytes 8bit 优化器
from bitsandbytes.optim import Adam8bit

# 检查 GPU 和 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 生成简单数据集 (y = 2*x + 3)
x = torch.linspace(-1, 1, 1000).unsqueeze(1)
y = 2 * x + 3 + 0.1 * torch.randn_like(x)  # 加点噪声
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# 简单线性模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleNet().to(device)

# 使用 bitsandbytes 8bit Adam
optimizer = Adam8bit(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 训练循环
for epoch in range(5):  # 测试用 5 个 epoch
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

print("训练完成！")
