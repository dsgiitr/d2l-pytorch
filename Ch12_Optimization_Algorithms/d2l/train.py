import numpy as np
import math
import time

from .base import try_gpu, Timer, Accumulator
from .figure import set_figsize, plt, Animator
from .data import data_iter_consecutive, data_iter_random
from .model import linreg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['evaluate_loss', 'train_ch10', 'train_2d','evaluate_accuracy', 'squared_loss', 'grad_clipping', 'sgd', 'train_and_predict_rnn', 'train_ch3', 'train_ch5','MaskedSoftmaxCELoss','train_ch7', 'translate_ch7', 'to_onehot' , 'predict_rnn', 'train_and_predict_rnn_nn', 'predict_rnn_nn', 'grad_clipping_nn']

def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset"""
    metric = Accumulator(2)  # sum_loss, num_examples
    for X, y in data_iter:
        metric.add(loss(net(X), y).sum().detach().numpy().item(), list(y.shape)[0])
    return metric[0] / metric[1]

def evaluate_accuracy(data_iter, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    net.eval()  # Switch to evaluation mode for Dropout, BatchNorm etc layers.
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0]
    return acc_sum.item()/n

def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.view(y_hat.shape)).pow(2) / 2

def grad_clipping(params, theta, device):
    """Clip the gradient."""
    norm = torch.tensor([0], dtype=torch.float32, device=device)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data.mul_(theta / norm)

def grad_clipping_nn(model, theta, device):
    """Clip the gradient for a nn model."""
    grad_clipping(model.parameters(), theta, device)

def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          corpus_indices, vocab, device, is_random_iter,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, prefixes):
    """Train an RNN model and predict the next item in the sequence."""
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss =  nn.CrossEntropyLoss()
    start = time.time()
    for epoch in range(num_epochs):
        if not is_random_iter:
            # If adjacent sampling is used, the hidden state is initialized
            # at the beginning of the epoch
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n = 0.0, 0
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:
                # If random sampling is used, the hidden state is initialized
                # before each mini-batch update
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                # Otherwise, the detach function needs to be used to separate
                # the hidden state from the computational graph to avoid
                # backpropagation beyond the current sample
                for s in state:
                    s.detach_()
            inputs = to_onehot(X, len(vocab))
            # outputs is num_steps terms of shape (batch_size, len(vocab))
            (outputs, state) = rnn(inputs, state, params)
            # After stitching it is (num_steps * batch_size, len(vocab))
            outputs = torch.cat(outputs, dim=0)
            # The shape of Y is (batch_size, num_steps), and then becomes
            # a vector with a length of batch * num_steps after
            # transposition. This gives it a one-to-one correspondence
            # with output rows
            y = Y.t().reshape((-1,))
            # Average classification error via cross entropy loss
            l = loss(outputs, y.long()).mean()
            l.backward()
            with torch.no_grad():
                grad_clipping(params, clipping_theta, device)  # Clip the gradient
                sgd(params, lr, 1)
            # Since the error is the mean, no need to average gradients here
            l_sum += l.item() * y.numel()
            n += y.numel()
        if (epoch + 1) % 50 == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            start = time.time()
        if (epoch + 1) % 100 == 0:
            for prefix in prefixes:
                print(' -',  predict_rnn(prefix, 50, rnn, params,
                                         init_rnn_state, num_hiddens,
                                         vocab, device))

def train_ch3(net, train_iter, test_iter, criterion, num_epochs, batch_size, lr=None):
    """Train and evaluate a model with CPU."""
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            optimizer.zero_grad()

            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            y = y.type(torch.float32)
            train_l_sum += loss.item()
            train_acc_sum += torch.sum((torch.argmax(y_hat, dim=1).type(torch.FloatTensor) == y).detach()).float()
            n += list(y.size())[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'\
            % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr=None):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        net.train() # Switch to training mode
        n, start = 0, time.time()
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device) 
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net, device) 
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'\
            % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc, time.time() - start))

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        weights = torch.ones_like(label)
        weights = SequenceMask(weights, valid_length).float()
        self.reduction='none'
        output=super(MaskedSoftmaxCELoss, self).forward(pred.transpose(1,2), label)
        return (output*weights).mean(dim=1)


def train_ch7(model, data_iter, lr, num_epochs, device): 
    """Train an encoder-decoder model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    tic = time.time()
    for epoch in range(1, num_epochs+1):
        l_sum, num_tokens_sum = 0.0, 0.0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_vlen, Y, Y_vlen = [x.to(device) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:,:-1], Y[:,1:], Y_vlen-1
            Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)
            l = loss(Y_hat, Y_label, Y_vlen).sum()
            l.backward()
            with torch.no_grad():
                grad_clipping_nn(model, 5, device)
            num_tokens = Y_vlen.sum().item()
            optimizer.step()
            l_sum += l.sum().item()
            num_tokens_sum += num_tokens
        if epoch % 50 == 0:
            print("epoch {0:4d},loss {1:.3f}, time {2:.1f} sec".format( 
                  epoch, (l_sum/num_tokens_sum), time.time()-tic))
            tic = time.time()

def translate_ch7(model, src_sentence, src_vocab, tgt_vocab, max_len, device):
    """Translate based on an encoder-decoder model with greedy search."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')]
    src_len = len(src_tokens)
    if src_len < max_len:
        src_tokens += [src_vocab.pad] * (max_len - src_len)
    enc_X = torch.tensor(src_tokens, device=device)
    enc_valid_length = torch.tensor([src_len], device=device)
    # use expand_dim to add the batch_size dimension.
    enc_outputs = model.encoder(enc_X.unsqueeze(dim=0), enc_valid_length)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_length)
    dec_X = torch.tensor([tgt_vocab.bos], device=device).unsqueeze(dim=0)
    predict_tokens = []
    for _ in range(max_len):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # The token with highest score is used as the next time step input.
        dec_X = Y.argmax(dim=2)
        py = dec_X.squeeze(dim=0).int().item()
        if py == tgt_vocab.eos:
            break
        predict_tokens.append(py)
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))


def to_onehot(X,size):
    return F.one_hot(X.long().transpose(0,-1), size)

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab, device):
    """Predict next chars with an RNN model"""
    state = init_rnn_state(1, num_hiddens, device)
    output = [vocab[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # The output of the previous time step is taken as the input of the
        # current time step.
        X = to_onehot(torch.tensor([output[-1]], dtype=torch.float32, device=device), len(vocab))
        # Calculate the output and update the hidden state
        (Y, state) = rnn(X, state, params)
        # The input to the next time step is the character in the prefix or
        # the current best predicted character
        if t < len(prefix) - 1:
            # Read off from the given sequence of characters
            output.append(vocab[prefix[t + 1]])
        else:
            # This is maximum likelihood decoding. Modify this if you want
            # use sampling, beam search or beam sampling for better sequences.
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([vocab.idx_to_token[i] for i in output])

def predict_rnn_nn(prefix, num_chars, batch_size, num_hiddens, num_layers, model, vocab, device):
    """Predict next chars with a RNN model."""
    # Use the model's member function to initialize the hidden state
    state = model.begin_state(num_hiddens=num_hiddens, device=device, num_layers=num_layers)
    output = [vocab[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], dtype=torch.float32, device=device).reshape((1, 1))
        # Forward computation does not require incoming model parameters
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(vocab[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([vocab.idx_to_token[i] for i in output])

def train_and_predict_rnn_nn(model, num_hiddens, init_gru_state, corpus_indices, vocab,
                                device, num_epochs, num_steps, lr,
                                clipping_theta, batch_size, prefixes, num_layers=1):
    """Train a RNN model and predict the next item in the sequence."""
    loss =  nn.CrossEntropyLoss()
    optm = torch.optim.SGD(model.parameters(), lr=lr)
    start = time.time()
    for epoch in range(1, num_epochs+1):
        l_sum, n = 0.0, 0
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps, device)
        state = model.begin_state(batch_size=batch_size, num_hiddens=num_hiddens, device=device ,num_layers=num_layers)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            X = X.to(dtype=torch.long)
            (output, state) = model(X, state)
            y = Y.t().reshape((-1,))
            l = loss(output, y.long()).mean()
            optm.zero_grad()
            l.backward(retain_graph=True)
            with torch.no_grad():
                # Clip the gradient
                grad_clipping_nn(model, clipping_theta, device)
                # Since the error has already taken the mean, the gradient does
                # not need to be averaged
                optm.step()
            l_sum += l.item() * y.numel()
            n += y.numel()

        if epoch % (num_epochs // 4) == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch, math.exp(l_sum / n), time.time() - start))
            start = time.time()
        if epoch % (num_epochs // 2) == 0:
            for prefix in prefixes:
                print(' -', predict_rnn_nn(prefix, 50, batch_size, num_hiddens, num_layers, model, vocab, device))

def train_2d(trainer):
    """Optimize a 2-dim objective function with a customized trainer."""
    # s1 and s2 are internal state variables and will 
    # be used later in the chapter
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def train_ch10(trainer, hyperparams, data_iter, feature_dim, num_epochs=2):
    # Initialization
    w1 = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b1 = np.zeros(1)
    w = Variable(torch.from_numpy(w1), requires_grad=True)
    b = Variable(torch.from_numpy(b1), requires_grad=True)

    if trainer.__name__ == 'SGD':
        optimizer = trainer([w, b], lr=hyperparams['lr'], momentum=hyperparams['momentum'])
    elif trainer.__name__ == 'RMSprop':
        optimizer = trainer([w, b], lr=hyperparams['lr'], alpha=hyperparams['gamma'])

    net, loss = lambda X: linreg(X, w, b), squared_loss
    # Train
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
            X, y = Variable(X), Variable(y)
            optimizer.zero_grad()
            output = net(X)
            l = loss(output, y).mean()
            l.backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             evaluate_loss(net, data_iter, loss))
                timer.start()
    print('loss: %.3f, %.3f sec/epoch'%(animator.Y[0][-1], timer.avg()))
    # return timer.cumsum(), animator.Y[0]