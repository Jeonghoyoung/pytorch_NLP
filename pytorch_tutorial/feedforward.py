import torch
import torch.nn as nn
# x가 2차원 행렬이라고 가정.

def linear(x, W, b):
    y = torch.mm(x,W) + b
    return y


# nn.Module
class MyLinear(nn.Module):
    '''
    parameters() 함수는 모듈내에 선언된 학습이 필요한 파라미터들을 반환하는 이터레이터
    ```python3

    params = [p.size() for p in linear.parameters()]

    ```

    '''

    def __int__(self, input_size, output_size):
        super().__init__()

        self.W = nn.Parameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(output_size), requires_grad=True)

    def forward(self, x):
        y = torch.mm(x, self.W) + self.b
        return y


if __name__ == '__main__':
    x = torch.FloatTensor(16, 10)
    W = torch.FloatTensor(10, 5)
    b = torch.FloatTensor(5)

    y = linear(x, W, b)
    print(y.shape)

