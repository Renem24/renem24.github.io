---
created: 2024-10-17 11:35
modified: 2024-10-17 11:35
---
# GD(Gradient Descent)
---
#optimization
## What is 'GD(Gradient Descent)'?
gradient의 반대 방향으로 가중치를 update하는 loss function 최적화 알고리즘

## How?
loss function의 gradient를 계산하고, 그 반대 방향으로 weight을 update

$$w = w - \eta \cdot \nabla L(w)$$