import torch

# Autograd : 자동으로 미분 및 역전파를 수행하는 기능
# 텐서들 간에 연산을 수행할 때 마다 동적으로 연산 그래프를 생성하여 연산의 결과물이 어떤 텐서로부터 어떤 연산을 통해서 왔는지 추적한다.

x = torch.FloatTensor(2,2)
y = torch.FloatTensor(2,2)
y.requires_grad_(True)

with torch.no_grad():
    z = (x+y) + torch.FloatTensor(2,2)
    print(z)


