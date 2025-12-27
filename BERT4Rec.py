import torch
from torch import nn

class GeGLU(nn.Module):
    """Функция активации GeGLU (GeLU/Swish + GLU).

    d_model: эмбеддинг модели (должен быть одинаков с encoder_layer).

    ff_dim: feed forward размерность.

    activation: функция активации GeLU можно заменить на Swish.

    approximate_gelu_tanh - аппроксимация функции GeLU по функции Tanh"""
    def __init__(self, d_model: int, ff_dim: int, activation: str = 'gelu', approximate_gelu_tanh: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim*2)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.glu = nn.GLU()

        if approximate_gelu_tanh:
            self.activation = nn.GELU(approximate='tanh')
        else:
            self.activation = nn.GELU()

        if activation == 'swish':
            self.activation = nn.Hardswish()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.glu(x)
        x = self.linear2(x)
        return x

class BERT4Rec(nn.Module):
    """Модель BERT4Rec

    user_id_count - число уникальных пользователей в датасете.

    movie_count - число уникальных фильмов.

    occupations - число уникальных занятостей людей (студент, художник и т.д.).

    genres - число уникальных жанров.

    emb_dim - размерность эмбеддингов.

    n_heads - количество голов внимания.

    n_layers - число повторения трансформерных блоков.

    activation - функция активации в энкодере трансформера.

    dropout - вероятность дропа в модулях внимания.

    dim_ff - размерность в feed forward слое."""
    def __init__(self,movie_count: int, emb_dim: int = 32, n_heads: int = 2, n_layers: int = 1, activation: str = 'gelu', dropout: float = 0.1, dim_ff: int = 768):
        super().__init__()
        self.movies_emb = nn.Embedding(num_embeddings=movie_count, embedding_dim=emb_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, batch_first=True, dropout=dropout, activation=activation, dim_feedforward=dim_ff)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_layers)
        self.act_func = GeGLU(d_model=emb_dim, ff_dim=dim_ff)

        self.final_projection = nn.Linear(emb_dim, movie_count)

    def forward(self, user, movie, genres, occupation, age, rating):
        movies = self.movies_emb(movie)

        x = self.transformer_encoder(movies)
        x = x[:, -1, :]
        x = self.act_func(x)
        return self.final_projection(x)