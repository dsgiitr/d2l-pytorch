
import torch
import torch.optim as optim


def evaluate_accuracy(data_iter, net):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = torch.Tensor([0]), 0
    for batch in data_iter:
        features, labels = batch
        for X, y in zip(features, labels):
            y = y.type(torch.FloatTensor)
            acc_sum += torch.sum((torch.argmax(net(X), dim=1).type(torch.FloatTensor) == y).detach()).float()
            n += 1
    return acc_sum.item() / n


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
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
