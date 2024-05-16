import torch.nn as nn


class AlertEmbeddingSimple(nn.Embedding):
    """
    表明导弹告警的方位（高低和方位）， 0 表示没有告警
    """

    def __init__(self, num_embeddings=101, embed_size=16):
        super().__init__(num_embeddings, embed_size)