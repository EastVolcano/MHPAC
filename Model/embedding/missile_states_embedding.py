import torch.nn as nn


class MissileStateEmbedding(nn.Embedding):
    """
    表明导弹的状态， 0 未发射 1 飞行中 2 命中 3 失效
    """

    def __init__(self, embed_size=16):
        super().__init__(4, embed_size)