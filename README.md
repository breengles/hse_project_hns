# hse_project_hns

Логическая последовательность реализаций по "сложности":

1. [MountainCar-v0_DASS](/MountainCar-v0_DASS)

   Табличный Q Learning с простой дискретизаций состояний.

2. [MountainCar-v0](/MountainCar-v0)

   Решение классической задачи с машинкой с помощью DQN, состояния непрерывные.

3. [MountainCarContinuous-v0_DAS](/MountainCarContinuous-v0_DAS)

   Решение задачи с машинкой с множеством непрерывных действий с помощью DQN, действия дискретизируются по "корзинкам"

4. [MountainCarContinuous-v0](/MountainCarContinuous-v0)

   Решение задачи с машинкой с множеством непрерывных действий с помощью DDPG.

   Нет reward shaping'а.

Note: в 1-3 сделан reward shaping с потенциальным изменением награды
