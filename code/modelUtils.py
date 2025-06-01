import copy
import torch
from torch import nn, optim, tensor

class ZhZhAutoencoder(nn.Module):
    """
    Autoencoder to learn the weights connecting Chinese embeddings back to one-hot vectors. 
    Initialize the weights connecting the embeddings and one-hot vectors with the transpose 
    of the weights from the Tecent pretrained embedding matrix, connecting the one-hot vector 
    and embedding, to speed up training.
    """
    def __init__(self, pretrained):
        super(ZhZhAutoencoder, self).__init__()

        # Save the pretrained embedding within the model
        # load pretrained embeddings and freeze them
        self.weights    = torch.FloatTensor(pretrained.vectors)
        self.encoder    = nn.Embedding.from_pretrained(self.weights, freeze = True)

        # Make a copy of the pretrained weights and use them in the following layers
        copy_pretrained = copy.deepcopy(pretrained)
        self.copyweight = torch.FloatTensor(copy_pretrained.vectors)
        self.decoder    = nn.Parameter(self.copyweight.t())

    def forward(self, text):
        """The pipeline that takes input values through the network"""
        # Find the embeddings for text
        trained_embed = self.encoder(text)

        # Turn embedding back to one-hot
        one_hot       = trained_embed @ self.decoder

        return one_hot
    
class EnZhEncoderDeocder(nn.Module):
    """
    Encoder Decoder to translate English words to Chinese words
    """
    
    def __init__(self, pretrained_en, pretrained_zh):
        super(EnZhEncoderDeocder, self).__init__()

        # Save the pretrained English embedding from Tencent within the model
        # load pretrained embeddings and freeze them
        self.weights_en = torch.FloatTensor(pretrained_en.vectors)
        self.encoder    = nn.Embedding.from_pretrained(self.weights_en, freeze = True)

        # Load the pretrained weights from zhzhautoencoder
        # Use the pretrained weights to map Chinese embedding to one-hot
        # ***** use the tranpose!!!
        self.zh_weights = pretrained_zh
        self.decoder    = nn.Parameter(self.zh_weights, requires_grad=False)
        self.bias       = nn.Parameter(torch.FloatTensor(1, 1).uniform_(-1, 1))

        # Hidden layer definition
        self.hidden = nn.Sequential(
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 200)
        )

    def forward(self, text):
        """The pipeline that takes input values through the network"""
        # Find the English embedding
        eng_emb = self.encoder(text)
        # Pass the English embedding through hidden layer
        zh_emb  = self.hidden(eng_emb) # torch.Size([256, 100])
        # Find the corresponding index of Chinese word for
        # the English embedding decoder: torch.Size([100, 2000000])
        one_hot = zh_emb @ self.decoder
        return one_hot + self.bias