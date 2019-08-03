"""The model module contains neural network building blocks"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['corr2d', 'linreg', 'RNNModel' , 'Encoder', 'Decoder', 'EncoderDecoder']

def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def linreg(X, w, b):
	"""Linear regression."""
	return torch.mm(X,w) + b

class RNNModel(nn.Module):
    """RNN model."""

    def __init__(self, rnn_layer, num_inputs, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.Linear = nn.Linear(num_inputs, vocab_size)

    def forward(self, inputs, state):
        """Forward function"""
        X = F.one_hot(inputs.long().transpose(0,-1), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.Linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, num_hiddens, device, batch_size=1, num_layers=1):
        """Return the begin state"""
        if num_layers == 1:
          return  torch.zeros(size=(1, batch_size, num_hiddens), dtype=torch.float32, device=device)
        else:
          return (torch.zeros(size=(1, batch_size, num_hiddens), dtype=torch.float32, device=device),
                  torch.zeros(size=(1, batch_size, num_hiddens), dtype=torch.float32, device=device))
        
class Residual(nn.Module):
  
  def __init__(self,input_channels, num_channels, use_1x1conv=False, strides=1, **kwargs):
    super(Residual, self).__init__(**kwargs)
    self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
    self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    if use_1x1conv:
      self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
    else:
      self.conv3 = None
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)
    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, X):
    
    Y = self.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))
    if self.conv3:
      X = self.conv3(X)
    Y += X
    Y =self.relu(Y)
    return Y

class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        """Forward function"""
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder archtecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        """Return the begin state"""
        raise NotImplementedError

    def forward(self, X, state):
        """Forward function"""
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """Forward function"""
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

