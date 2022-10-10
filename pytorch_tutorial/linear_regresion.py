import random
import torch
import torch.nn as nn
'''

임의의 함수를 근사하는 회귀분석 예제 코드

1. 임의로 텐서 생성
2. 텐서들을 근사하고자 하는 정답 함수에 넣어 y를 구하기
3. 그 y와 신경망을 통과한 y_hat 과의 error를 MSE를 통하여 구하기
4. 확률적 경사하강법 (SGD)를 통해 최적화

'''

'''
딥러닝 수행과정

1. nn.Module 클래스를 상속받아 모델 아키텍처 클래스 선언
2. 해당 클래스 객체생성
3. SGD, Adam 등의 옵티마이저를 생성하고 생성한 모델의 파라미터를 최적화 대상으로 등록
4. 데이터로 미니매치를 구성하여 피드포워드 연산 그래프 생성
5. 손실 함수를 통해 최종 결괏값(scalar)과 손실값(loss) 계산
6. 손실에 대해서 backward() 호출 -> 연산 그래프 상의 텐서들의 기울기가 채워짐
7. 3번의 옵티마이저에서 step()을 호출하여 경사하강법 1 스텝 수행
8. 4번으로 돌아가 수렴 조건이 만족할 때까지 반복 수행

'''


class Mymodel(nn.Module):

    def __init__(self, input_size, output_size):
        super(Mymodel, self).__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)
        return y


def ground_truth(x):
    return 3 * x[:, 0] + x[:, 1] - 2 * x[:, 2]


def train(model, x, y, optim):
    # initialize gradients in all parameters in module
    optim.zero_grad()

    # feed-forward
    y_hat = model(x)

    # get error between answer and inferenced
    loss = ((y - y_hat)**2).sum() / x.size(0)

    # back-propagation
    loss.backward()

    # one-step of gradient descent
    optim.step()

    return loss.data


if __name__ == '__main__':
    #params
    # epochs = 전체 데이터셋에 대해 한 번 학습을 완료한 상태
    # batch size = 한번의 배치마다 주는 데이터 샘플의 사이즈
    # batch = 나눠진 데이터 셋을 의미.
    # iteration = 1 epoch을 마치는데 필요한 미니배치 갯수를 의미한다. 즉 1 epoch을 마치는데 필요한 파라미터 업데이트 횟수를 의미
    batch_size = 1
    n_epochs = 1000
    n_iter = 10000

    model = Mymodel(3,1)

    optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.1)
    print(model)

    for e in range(n_epochs):
        avg_loss = 0

        for i in range(n_iter):
            x = torch.rand(batch_size, 3)
            y = ground_truth(x.data)

            loss = train(model, x, y, optim)

            avg_loss += loss
            avg_loss = avg_loss / n_iter

        # simple test sample to check the network
        x_valid = torch.FloatTensor([[.3, .2, .1]])
        y_valid = ground_truth(x_valid.data)

        model.eval()
        y_hat = model(x_valid)
        model.train()

        print(avg_loss, y_valid.data[0], y_hat.data[0,0])

        if avg_loss < .001:
            break

