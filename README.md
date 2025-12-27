# RecomenderSystem-based-on-NeuralNetworks-comaprison
Первый проект по машинному обучению. Сравнение архитектур нейросетей(LSTM, LSTM и Multi Head Attention, BERT) для рекомендательных систем.

Ниже представлены сравнения потерь и метрик точности моделей с разными параметрами и параметры тренировки моделей.

initial_lr = 0.003 (Первоначальная скорость обучения)
min_lr = 0.003 (Минимальная скорость обучения)
weight_decay = 0.01 (Регуляризация весов)
emb_dim = 32 (Размерность эмбеддингов)
n_heads = 2 (Количество голов внимания)
n_layers = 1 (Количество повторений блоков LSTM/BERT)
dropout = 0.1 (Регуляризация)
hidded_dim = 64 (Скрытый слой LSTM)
dim_ff = 128 (Feed Forward размерность у BERT)
batch_size = 256 (Размер пакета)
train_ratio = 0.9 (Процент тренировочный данных)
seq_len_for_training = 50 (Длинна последовательности)
top_k_for_training = 10 (Топ рекомендаций)
top_k_for_reccomending = 10 (Топ рекомендаций)
training_epoch = 20 (Количество эпох тренировки)
eval_freq = 1000 (Частота оценки в шагах)
plot_results = True (Вывести резульататы тренировки)

LSTM модель:
<img width="1000" height="500" alt="LSTM CE loss" src="https://github.com/user-attachments/assets/f88dd2ab-ed07-440c-a941-4f1fde014829" />
<img width="1000" height="500" alt="LSTM Accuracy metrics" src="https://github.com/user-attachments/assets/4bbe46aa-4aee-4206-a417-4a76a667fcec" />



