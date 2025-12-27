import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from BERT4Rec import GeGLU, BERT4Rec
from LSTMs import LSTMModel, LSTMattnModel
import matplotlib.pyplot as plt
import seaborn as sns
import time



movies = pd.read_csv('E:/ReccomenderSysComparison/data/movies.dat',sep="::", engine='python',
                      names=["movie_id", "title", "genres"], encoding='latin-1')

users = pd.read_csv('E:/ReccomenderSysComparison/data/users.dat',sep="::", engine='python',
                      names=["user_id", "gender", "age", "occupation",'zip'], encoding='latin-1')

ratings = pd.read_csv('E:/ReccomenderSysComparison/data/ratings.dat',sep="::", engine='python',
                      names=["user_id", "movie_id", "rating", "timestamp"], encoding='latin-1')

plot_pop_bias = False
if plot_pop_bias:
    count = ratings['movie_id'].value_counts()
    sns.barplot(data=count.values)
    plt.title('Popularity bias')
    plt.ylabel('Popularity')
    plt.xlabel('Movies')
    plt.show()

occupation_encode = {
    0: "other or not specified", 1: "academic/educator", 2: "artist",
    3: "clerical/admin", 4: "college/grad student", 5: "customer service",
    6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer",
    12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}
a = dict(zip(movies['movie_id'], movies['title']))
movie_decoder = {g: i for g, i in a.items()}



def preprocess_datasets(movies: pd.DataFrame, users: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Обработка датасета"""

    genres = sorted(set('|'.join(movies['genres']).split('|')))
    genres_encode = {g: i + 1 for i, g in enumerate(genres)}

    movies['genres'] = movies['genres'].apply(lambda x: [genres_encode[b] for b in x.split('|')])

    movies.drop(columns=['title'], inplace=True)

    merged_data = pd.merge(
        ratings,
        users,
        on='user_id',
        how='inner'
    )
    merged_data.drop(columns=['zip', 'timestamp'], inplace=True)

    full_data = pd.merge(
        merged_data,
        movies,
        on='movie_id',
        how='inner'
    )

    full_data['movie_id'] = full_data['movie_id'].apply(lambda x: x - 1)
    full_data['rating'] = full_data['rating'].apply(lambda x: 1 if x >= 4 else 0)

    col = ['gender']
    full_data = pd.get_dummies(full_data, columns=col, drop_first=True)
    full_data['gender_M'] = full_data['gender_M'].astype(int)

    padded_genres = []
    for i in full_data['genres']:
        current_length = len(i)
        pad_len = 6 - current_length
        pad_seq = [0] * pad_len + i
        padded_genres.append(pad_seq)
    full_data['genres'] = padded_genres

    return full_data

#dataset info for training
full_data = preprocess_datasets(movies, users, ratings)
user_id_count = full_data['user_id'].nunique()
movie_count = 3990
occupation_count = full_data['occupation'].nunique()
genres_count = 18



class ValPyDataset(Dataset):
    """Создание проверочного датасета методом LOO (Leave One Out)"""
    def __init__(self, data: pd.DataFrame, seq_len: int, val_ratio: float):
        self.samples = []
        groups = data.groupby('user_id')

        for user_id, user_data in groups:
            movies = user_data['movie_id'].tolist()
            genres = user_data['genres'].tolist()
            ratings = user_data['rating'].tolist()

            user_info = user_data.iloc[0]

            if len(movies) < seq_len + 2:
                continue

            split_point = int(len(movies) * val_ratio)
            if split_point < seq_len:
                continue

            for i in range(split_point, len(movies) - seq_len):
                sample = {
                    'user': torch.tensor(user_id),
                    'movies_seq': torch.tensor(movies[i:i + seq_len]),
                    'genres': torch.tensor(genres[i:i + seq_len]),
                    'occupation': torch.tensor(user_info['occupation']),
                    'age': torch.tensor(user_info['age']),
                    'rating': torch.tensor(ratings[i:i + seq_len]),
                    'validation_target': torch.tensor(movies[i + seq_len])
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TrainPyDataset(Dataset):
    """Тренировочный датасет"""
    def __init__(self, data: pd.DataFrame, seq_len: int, train_ratio: float):
        self.samples = []
        groups = data.groupby('user_id')

        for user_id, user_data in groups:
            movies = user_data['movie_id'].tolist()
            genres = user_data['genres'].tolist()
            ratings = user_data['rating'].tolist()

            if len(movies) < seq_len + 1:
                continue

            user_info = user_data.iloc[0]

            split_point = int(len(movies) * train_ratio)

            for i in range(split_point - seq_len):
                if i + seq_len >= split_point:
                    break

                sample = {
                    'user': torch.tensor(user_id),
                    'movies_seq': torch.tensor(movies[i:i + seq_len]),
                    'genres': torch.tensor(genres[i:i + seq_len]),
                    'occupation': torch.tensor(user_info['occupation']),
                    'age': torch.tensor(user_info['age']),
                    'rating': torch.tensor(ratings[i:i + seq_len]),
                    'target': torch.tensor(movies[i + seq_len])
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]




class FormatReccomend(Dataset):
    """Форматирование данных пользователя в нужный вид для системы"""
    def __init__(self, data: pd.DataFrame):
        self.user_sample = []

        group_by = data.groupby('user_id')

        for user, user_data in group_by:
            movies = user_data['movie_id'].tolist()
            genres = user_data['genres'].tolist()
            ratings = user_data['rating'].tolist()

            user_info = user_data.iloc[0]

            sample = {
                "user_id": torch.tensor(user),
                "movie_id": torch.tensor(movies),
                "genres": torch.tensor(genres),
                "occupation": torch.tensor(user_info['occupation']),
                "age": torch.tensor(user_info['age']),
                "rating": torch.tensor(ratings)
            }

            self.user_sample.append(sample)


    def __len__(self):
        return len(self.user_sample)


    def __getitem__(self, item):
        return self.user_sample[item]





class ExperementalReccomenderSystem:
    """Класс для тренировки и выдачи рекомендаций для пользователя"""

    def __init__(self, model, file_name_to_save: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.file_to_save = file_name_to_save
        self.device = device

    def training_model(self, epochs: int, optimizer: torch.optim, dataloader: DataLoader, dataloader_val: DataLoader, eval_freq: int, top_k: int, initial_lr: float, min_lr: float, device, crit, gradient_clip: bool):
        """Тренировка модели"""
        model.to(device)
        peak_lr = optimizer.param_groups[0]['lr']
        total_training_steps = len(dataloader) * epochs
        num_warmup_steps = int(total_training_steps * 0.1)
        lr_increment = (peak_lr - initial_lr) / num_warmup_steps
        train_loss_df, val_loss_df, hitrate_df, ndcg_df = [], [], [], []

        for epoch in range(epochs):
            train_loss_info, val_loss_info = [], []

            global_step = 0

            for batch in dataloader:
                self.model.train()
                global_step += 1

                optimizer.zero_grad()

                if global_step < num_warmup_steps:
                    lr = initial_lr + global_step * lr_increment
                else:
                    progress = ((global_step - num_warmup_steps) / (total_training_steps - num_warmup_steps))
                    lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

                for param in optimizer.param_groups:
                    param['lr'] = lr

                output = self.model(batch['user'].to(device), batch['movies_seq'].to(device), batch['genres'].to(device),
                               batch['occupation'].to(device), batch['age'].to(device),
                               batch['rating'].to(device))

                loss = crit(output, batch['target'].to(device))

                loss.backward()

                if gradient_clip:
                    if global_step > num_warmup_steps:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    else:
                        if global_step >= num_warmup_steps:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss_info.append(loss.item())
                train_loss_df.append({'loss': loss.item(), 'epoch': epoch+1})

                if global_step % eval_freq == 0:
                    hits = 0
                    attempt = 0
                    mean_hits = []
                    ndcgs = []
                    self.model.eval()
                    with torch.no_grad():
                        for vbatch in dataloader_val:
                            val_output = self.model(vbatch['user'].to(device), vbatch['movies_seq'].to(device),
                                               vbatch['genres'].to(device), vbatch['occupation'].to(device),
                                               vbatch['age'].to(device),
                                               vbatch['rating'].to(device))

                            val_loss = crit(val_output, vbatch['validation_target'].to(device))
                            val_loss_info.append(val_loss)
                            val_loss_df.append({'val_loss': val_loss.item(), 'epoch': epoch+1})

                            output_for_hitrate = torch.softmax(val_output, dim=1)
                            topk = torch.topk(output_for_hitrate, k=top_k).indices

                            for preds, true in zip(topk, vbatch['validation_target']):
                                hits += 1 if true in preds else 0
                                attempt += 1
                                mean_hits.append(hits / attempt)
                                if true in preds:
                                    idx = (preds == true).nonzero(as_tuple=True)[0].item()
                                    ndcgs.append(1 / np.log2(idx + 2))
                                else:
                                    ndcgs.append(0)

                    print(f'Hitrate@{top_k}: {np.mean(mean_hits)} NDGC@{top_k}: {np.mean(ndcgs)}')
                    print(f'eval_loss CE: {torch.mean(torch.tensor(val_loss_info))}')

                    hitrate_df.append({f'hitrate':np.mean(mean_hits), 'epoch': epoch+1})
                    ndcg_df.append({f'ndcg':np.mean(ndcgs), 'epoch': epoch+1})


                else:
                    print(
                        f'mean_train_loss CE: {torch.mean(torch.tensor(train_loss_info))} -- step: {global_step} -- Epoch {epoch+1}')

        torch.save(self.model.state_dict(), f'{self.file_to_save}.pth')

        return pd.DataFrame(train_loss_df), pd.DataFrame(val_loss_df), pd.DataFrame(hitrate_df), pd.DataFrame(ndcg_df)


    def reccomend(self, file: str, user_id: int, ids_of_watched_films: list[int], liked: list[int], occupation: int, age_group: int, k: int):
        """Получить рекомендации для пользователя
        Возвращает топ рекомендованных фильмов"""
        self.model.load_state_dict(torch.load(file, weights_only=True))
        self.model.eval()
        self.model.to(self.device)

        liked = [1 if x>=4 else 0 for x in liked]
        films_info = movies.iloc[ids_of_watched_films]
        films_info = pd.DataFrame(films_info)
        films_info.reset_index(inplace=True, drop=True)

        padded_genres = []
        for i in films_info['genres']:
            current_length = len(i)
            pad_len = 6 - current_length
            pad_seq = [0] * pad_len + i
            padded_genres.append(pad_seq)
        films_info['genres'] = padded_genres

        occupation = [occupation] * len(ids_of_watched_films)
        age_group = [age_group] * len(ids_of_watched_films)
        user_id = [user_id] * len(ids_of_watched_films)

        films_info = pd.concat(
            [pd.DataFrame(user_id, columns=['user_id']), films_info, pd.DataFrame(liked, columns=['rating']),
             pd.DataFrame(occupation, columns=['occupation']),
             pd.DataFrame(age_group, columns=['age'])], axis=1)
        a = FormatReccomend(films_info)
        loader = DataLoader(a)

        with torch.no_grad():
            for data in loader:
                output = self.model(data['user_id'].to(self.device), data['movie_id'].to(self.device), data['genres'].to(self.device),
                               data['occupation'].to(self.device), data['age'].to(self.device), data['rating'].to(self.device))

                softmax_output = torch.softmax(output, dim=1)
                topk_output = torch.topk(softmax_output, sorted=True, k=k).indices.tolist()[0]
                decoded_output = [movie_decoder[x] for x in topk_output]
                print(f'Watched films: {[movie_decoder[x] for x in ids_of_watched_films]}')
            return decoded_output


class TrainingConfig:
    """Численные параметры тренировки"""

    def __init__(self,initial_lr: float = 0.003, min_lr: float = 0.003, weight_decay: float = 0.01, emb_dim: int = 32, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = 0.1, hidded_dim: int = 64, dim_ff: int = 128, batch_size: int = 256, train_ratio: float = 0.9,
                 seq_len_for_training: int = 50, top_k_for_training: int = 10, top_k_for_reccomending: int = 10, training_epochs: int = 2,
                 eval_freq: int = 1000, plot_results: bool = True):

        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.dim_ff = dim_ff
        self.hidden_dim = hidded_dim
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.seq_len_training = seq_len_for_training
        self.top_k_training = top_k_for_training
        self.top_k_rec = top_k_for_reccomending
        self.training_epochs = training_epochs
        self.eval_frequency = eval_freq
        self.plot_results = plot_results


def plot_results(results):
    train_loss, val_loss, hitrate, ndcg = results
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_loss.groupby('epoch')['val_loss'].mean(), label="validation set")
    plt.plot(train_loss.groupby('epoch')['loss'].mean(), label="training set")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.figure(figsize=(10,5))
    plt.plot(hitrate.groupby('epoch')['hitrate'].mean(), label=f'Hitrate@{training_config.top_k_training}')
    plt.plot(ndcg.groupby('epoch')['ndcg'].mean(), label=f'NDCG@{training_config.top_k_training}')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy metrics")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    start = datetime.datetime.now()

    training_config = TrainingConfig()

    val = ValPyDataset(full_data, training_config.seq_len_training, val_ratio=training_config.train_ratio)
    train = TrainPyDataset(full_data, training_config.seq_len_training, train_ratio=training_config.train_ratio)
    train_loader = DataLoader(train, batch_size=training_config.batch_size, pin_memory=True)
    val_loader = DataLoader(val, batch_size=training_config.batch_size, pin_memory=True)

    model = LSTMattnModel(user_id_count, movie_count, occupation_count, genres_count, training_config.emb_dim, training_config.hidden_dim, training_config.dropout, training_config.n_heads, training_config.n_layers)
    print(f'Количество параметров модели: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.initial_lr, weight_decay=training_config.weight_decay, fused=True)
    criterion = torch.nn.CrossEntropyLoss()

    run_training = ExperementalReccomenderSystem(file_name_to_save='YOUR_FILE_NAME', model=model, device='cuda')
    #results = run_training.training_model(epochs=training_config.training_epochs, optimizer=optimizer, dataloader=train_loader, dataloader_val=val_loader, eval_freq=training_config.eval_frequency, top_k=training_config.top_k_training, initial_lr=training_config.initial_lr,
    #                                      min_lr=training_config.min_lr, device=run_training.device, crit=criterion, gradient_clip=True)
    #end = datetime.datetime.now()
    #print(f"Время тренировки: {end - start}")


    if training_config.plot_results:
        pass
        #plot_results(results)

    print(run_training.reccomend(file='YOUR_FILE_NAME', user_id=222, ids_of_watched_films=[1,22, 567, 1208, 2567, 3000, 560], liked=[4,4,3,3,5,3,5], age_group=10, occupation=5, k=training_config.top_k_rec))