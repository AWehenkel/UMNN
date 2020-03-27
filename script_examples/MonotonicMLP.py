import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from models.UMNN import MonotonicNN, IntegrandNN

def f(x_1, x_2, x_3):
    return .001*(x_1**3 + x_1) + x_2 ** 2 + torch.sin(x_3)

def create_dataset(n_samples):
    x = torch.randn(n_samples, 3)
    y = f(x[:, 0], x[:, 1], x[:, 2])
    return x, y

class MLP(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(MLP, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-nb_train", default=10000, type=int, help="Number of training samples")
    parser.add_argument("-nb_test", default=1000, type=int, help="Number of testing samples")
    parser.add_argument("-nb_epoch", default=200, type=int, help="Number of training epochs")
    parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
    parser.add_argument("-folder", default="", help="Folder")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_monotonic = MonotonicNN(3, [100, 100, 100], nb_steps=100, dev=device).to(device)
    model_mlp = MLP(3, [200, 200, 200]).to(device)
    optim_monotonic = torch.optim.Adam(model_monotonic.parameters(), 1e-3, weight_decay=1e-5)
    optim_mlp = torch.optim.Adam(model_mlp.parameters(), 1e-3, weight_decay=1e-5)

    train_x, train_y = create_dataset(args.nb_train)
    test_x, test_y = create_dataset(args.nb_test)
    b_size = 100

    for epoch in range(0, args.nb_epoch):
        # Shuffle
        idx = torch.randperm(args.nb_train)
        train_x = train_x[idx].to(device)
        train_y = train_y[idx].to(device)
        avg_loss_mon = 0.
        avg_loss_mlp = 0.
        for i in range(0, args.nb_train-b_size, b_size):
            # Monotonic
            x = train_x[i:i + b_size].requires_grad_()
            y = train_y[i:i + b_size].requires_grad_()
            y_pred = model_monotonic(x[:, [0]], x[:, 1:])[:, 0]
            loss = ((y_pred - y)**2).sum()
            optim_monotonic.zero_grad()
            loss.backward()
            optim_monotonic.step()
            avg_loss_mon += loss.item()
            # MLP
            y_pred = model_mlp(x[:, [0]], x[:, 1:])[:, 0]
            loss = ((y_pred - y) ** 2).sum()
            optim_mlp.zero_grad()
            loss.backward()
            optim_mlp.step()
            avg_loss_mlp += loss.item()

        print(epoch)
        print("\tMLP: ", avg_loss_mlp/args.nb_train)
        print("\tMonotonic: ", avg_loss_mon / args.nb_train)

    # <<TEST>>
    x = torch.arange(-5, 5, .1).unsqueeze(1).to(device)
    h = torch.zeros(x.shape[0], 2).to(device)
    y = f(x[:, 0], h[:, 0], h[:, 1]).detach().cpu().numpy()
    y_mon = model_monotonic(x, h)[:, 0].detach().cpu().numpy()
    y_mlp = model_mlp(x, h)[:, 0].detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    plt.plot(x, y_mon, label="Monotonic model")
    plt.plot(x, y_mlp, label="MLP model")
    plt.plot(x, y, label="groundtruth")
    plt.legend()
    plt.show()
    plt.savefig("Monotonicity.png")



