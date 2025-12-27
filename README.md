# RecomenderSystem-based-on-NeuralNetworks-comaprison
Первый проект по машинному обучению. мини сравнение архитектур нейросетей(LSTM, LSTM и Multi Head Attention, BERT) для рекомендательных систем.

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

LSTM + Multi Head Attention (2 головы внимания):
<img width="1000" height="500" alt="LSTM_MHA CE loss" src="https://github.com/user-attachments/assets/d0ca6304-8b6a-4d33-87ab-f68e88fe0794" />
<img width="1000" height="500" alt="LSTM_MHA accuracy metrics" src="https://github.com/user-attachments/assets/b0710ecd-36e5-4541-9916-ccda42a3346e" />

LSTM + Multi Head Attention (8 голов внимания):
<img width="1000" height="500" alt="LSTM_MHA 8 heads CE loss" src="https://github.com/user-attachments/assets/9a9bfdb4-11c1-4735-bc3a-94e2c4c104fb" />
<img width="1000" height="500" alt="LSTM_MHA 8 heads accuracy metrics loss" src="https://github.com/user-attachments/assets/ac08be25-d83c-410c-9bd2-877c32014a9d" />

BERT (2 головы внимания):
<img width="1000" height="500" alt="BERT 2 heads CE losses" src="https://github.com/user-attachments/assets/d2f69018-8f16-46f5-99a4-04dc2a3d58dd" />
<img width="1000" height="500" alt="BERT 2 heads accuracy metrics" src="https://github.com/user-attachments/assets/c3bd380a-cf2b-4c55-b45b-e9afb11ca4be" />

BERT (8 голов внимания):
<img width="1000" height="500" alt="BERT 8 heads CE losses" src="https://github.com/user-attachments/assets/4cdf02ad-0d6b-450d-8f39-76746081af79" />
<img width="1000" height="500" alt="BERT 8 heads accuracy metrics" src="https://github.com/user-attachments/assets/ec63797a-a3d8-40c8-8046-ff33cef7d05e" />


Лучше всего показала себя модель LSTM + Multi Head Attention с 2мя головами внимания, демонстрируя значительное превосходство перед другими архитектурами.
