---
created: 2024-09-24 10:16
modified: 2024-10-17 11:51
---
# EMA(Exponential Moving Average)
---
## What is 'EMA(Exponential Moving Average)'?

### Moving Average
일정 기간동안의, data의 평균을 계산하는 방법

### Exponential Moving Average
**최근 data**에 **더 큰 가중치**를 부여하는 Moving Average 방법

## How?

$$EMA_{t} = \alpha \cdot x_{t} + (1 - \alpha) \cdot EMA_{t-1}$$