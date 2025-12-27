import torch
from torch import nn


class LSTMModel(nn.Module):
    """Модель LSTM + MHA

           user_id_count - число уникальных пользователей в датасете.

           movie_count - число уникальных фильмов.

           occupations - число уникальных занятостей людей (студент, художник и т.д.).

           genres - число уникальных жанров.

           emb_dim - размерность эмбеддингов.

           hidden_dim - размерность скрытого состояния LSTM слоя.

           n_heads - количество голов внимания.

           n_layers_lstm - число повторения трансформерных блоков.

           activation - функция активации в энкодере трансформера.

           dropout - вероятность дропа в модулях внимания."""
    def __init__(self, user_id_count: int, movie_count: int, occupations: int, genres: int,emb_dim: int, hidden_dim: int, dropout: float, n_layers: int):
        super().__init__()
        self.user_emb = nn.Embedding(user_id_count+1, emb_dim)
        self.mov_emb = nn.Embedding(movie_count, emb_dim)
        self.occ_emb = nn.Embedding(occupations, emb_dim)
        self.genres_emb =nn.Embedding(genres+1, emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=emb_dim*2+1, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.final_projection = nn.Linear(hidden_dim+emb_dim*2+1, movie_count)

    def forward(self, user, movie, genres, occupation, age, rating):

        user = self.user_emb(user)
        movies = self.mov_emb(movie)
        genres = self.genres_emb(genres)
        occup = self.occ_emb(occupation)


        genres_emb_agg = torch.mean(genres, dim=2)

        seq_features_for_lstm = torch.cat([movies, genres_emb_agg, rating.unsqueeze(-1)], dim=2)

        lstm_out, (h0, c0) = self.lstm(seq_features_for_lstm)

        last_hidden = lstm_out[:, -1, :]



        user_context = torch.cat([user, occup, age.unsqueeze(-1)], dim=1)


        final_proj =  torch.cat([last_hidden, user_context], dim=1)


        final_proj = self.final_projection(final_proj)


        return final_proj


class LSTMattnModel(nn.Module):
    """Модель LSTM + MHA

        user_id_count - число уникальных пользователей в датасете.

        movie_count - число уникальных фильмов.

        occupations - число уникальных занятостей людей (студент, художник и т.д.).

        genres - число уникальных жанров.

        emb_dim - размерность эмбеддингов.

        hidden_dim - размерность скрытого состония LSTM слоя.

        n_heads - количество голов внимания.

        n_layers_lstm - число повторения трансформерных блоков.

        activation - функция активации в энкодере трансформера.

        dropout - вероятность дропа в модулях внимания."""
    def __init__(self, user_id_count: int, movie_count: int, occupations: int, genres: int, emb_dim: int, hidden_dim: int, dropout: float, n_heads: int, n_layers_lstm: int):
        super().__init__()
        self.user_emb = nn.Embedding(user_id_count+1, emb_dim)
        self.mov_emb = nn.Embedding(movie_count, emb_dim)
        self.occ_emb = nn.Embedding(occupations, emb_dim)
        self.genres_emb =nn.Embedding(genres+1, emb_dim, padding_idx=0)


        self.attn = nn.MultiheadAttention(hidden_dim,n_heads,dropout, batch_first=True)
        self.lstm = nn.LSTM(input_size=emb_dim*2+1, hidden_size=hidden_dim, num_layers=n_layers_lstm, batch_first=True)
        self.final_projection = nn.Linear(hidden_dim+emb_dim*2+1, movie_count)


    def forward(self, user, movie, genres, occupation, age, rating):

        user = self.user_emb(user)
        movies = self.mov_emb(movie)
        genres = self.genres_emb(genres)
        occup = self.occ_emb(occupation)


        genres_emb_agg = torch.mean(genres, dim=2)

        seq_features_for_lstm = torch.cat([movies, genres_emb_agg, rating.unsqueeze(-1)], dim=2)

        lstm_out, (h0, c0) = self.lstm(seq_features_for_lstm)

        last_hidden = lstm_out[:, -1, :]

        last_hidden_attn, _ = self.attn(last_hidden, last_hidden, last_hidden)

        user_context = torch.cat([user, occup, age.unsqueeze(-1)], dim=1)

        final_proj =  torch.cat([last_hidden_attn, user_context], dim=1)

        final_proj = self.final_projection(final_proj)

        return final_proj