# Training and test loops
import torch

def train_loop(dataloader, model, optimizer, max_grad_norm=5.0):
    """One training epoch.

    max_grad_norm clips the total gradient norm before each optimizer step; pass
    None to disable. Rational-quadratic splines occasionally produce very large
    gradients (for samples landing near a bin edge or the tail boundary), and a
    single unclipped spike can push the flow into a worse configuration it does
    not recover from within the run.
    """
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
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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