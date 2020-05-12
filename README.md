# lab5
## Step decay

На графиках ниже представлены:
Оранжевый - initial_lr = 0.001, drop = 0.3; Синий - initial_lr = 0.001, drop = 0.5; Красный -  initial_lr = 0.01, drop = 0.3; Голубой - Warm-Up 10 эпох, lr с 0.001 до 0.01, drop = 0.3.
Как видим по обучающей выборке, сильно выделяется красный: сошёлся хуже остальных, при этом имеет самый нестабильный характер, в то время как остальные три сошлись по точности уже к 10-15 эпохам и идут гладко. Оранжевый и синий ведут себя практически идентично, у красного же на валидационной выборке наблюдаются самые большие разбросы по точности, а сходимость по ошибке самая последняя. Оранжевый и синий сходятся в одно время как по ошибке, так и по точности на валидационной выборке, значит оптимальным вариантом будет явно с lr = 0.001, drop = 0.3/0.5.
![1 stage](stage1.jpg)

## Exponential Decay

На графиках ниже представлены: 
Розовый - lr = 0.01, k = 0.25; Зелёный - lr = 0.01, k = 0.5; Серый - lr = 0.001, k = 0.4; Оранжевый - warm-up с lr =  0.001 до 0.01, k = 0.25.
На графиках видим, что серый и оранжевый демонстрируют хорошую и достаточно быструю сходимость по точности на обучающей выборке, однако у серого есть некоторые колебания, в то время как оранжевый идёт плавно. Зелёный сошёлся не лучшим образом на фоне первый двух, розовый еще хуже. По ошибке на обучающей выборке лучше всех сходится серый и не проявляет так много выбросов, как оранжевый. На валидационных выборках розовый и зелёный имеют самые большие разбросы, в том время как серый и оранжевый ведут себя чуть более спокойно. Оптимальным вариантом в нашем случае будет серый с lr = 0.01, k = 0.5.
![2 stage](stage2.jpg)
