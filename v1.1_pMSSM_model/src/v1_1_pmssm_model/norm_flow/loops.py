# Training and test loops
import torch

def train_loop(dataloader, model, optimizer):

    model.train()

    size = len(dataloader.dataset)
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute negative log probability loss
        loss = - model.log_prob(y, X)

        train_loss += loss.detach().sum()
        loss = loss.mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d} samples]")

    average_loss = train_loss.item() / size
    print('Average loss: {:.4f}'.format(average_loss))
    return average_loss


def test_loop(dataloader, model):

    model.eval()

    size = len(dataloader.dataset)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            loss = - model.log_prob(y, X)
            test_loss += loss.sum()

    test_loss = test_loss.item() / size
    print(f"Test loss: {test_loss:>8f} \n")
    return test_loss


# Train/test loops for masking.MaskedFlow, whose log_prob takes a joint
# vector plus a per-example conditioning mask instead of (params, context).

def train_loop_masked(dataloader, model, optimizer):

    model.train()

    size = len(dataloader.dataset)
    train_loss = 0

    for batch, (v, mask) in enumerate(dataloader):
        loss = - model.log_prob(v, mask)

        train_loss += loss.detach().sum()
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(v)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d} samples]")

    average_loss = train_loss.item() / size
    print('Average loss: {:.4f}'.format(average_loss))
    return average_loss


def test_loop_masked(dataloader, model):

    model.eval()

    size = len(dataloader.dataset)
    test_loss = 0

    with torch.no_grad():
        for v, mask in dataloader:
            loss = - model.log_prob(v, mask)
            test_loss += loss.sum()

    test_loss = test_loss.item() / size
    print(f"Test loss: {test_loss:>8f} \n")
    return test_loss