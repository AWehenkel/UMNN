import torch
import argparse
import torch.nn as nn
from models.UMNN import MonotonicNN, IntegrandNN


def create_dataset(n_samples):
    x = torch.randn(n_samples, 3)
    y = x[:, 0]**3 + x[:, 1]**2 + torch.sin(x[:, 2])
    return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-nb_train", default=10000, type=int, help="Number of training samples")
    parser.add_argument("-nb_test", default=1000, type=int, help="Number of testing samples")
    parser.add_argument("-nb_epoch", default=100, type=int, help="Number of training epochs")
    parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
    parser.add_argument("-folder", default="", help="Folder")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MonotonicNN(3, [100, 100, 100], nb_steps=20, dev=device).to(device)
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

    train_x, train_y = create_dataset(args.nb_train)
    test_x, test_y = create_dataset(args.nb_test)
    b_size = 100

    for epoch in range(0, args.nb_epoch):
        # Shuffle
        idx = torch.randperm(args.nb_train)
        train_x = train_x[idx].to(device)
        train_y = train_y[idx].to(device)
        avg_loss = 0.
        for i in range(0, args.nb_train-b_size, b_size):
            x = train_x[i:i + b_size].requires_grad_()
            y = train_y[i:i + b_size].requires_grad_()
            y_pred = model(x[:, [0]], x[:, 1:])[:, 0]
            loss = ((y_pred - y)**2).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss += loss.item()
        print("train:", epoch, avg_loss / (i + b_size))
        avg_loss = 0.
        for i in range(0, args.nb_test-b_size, b_size):
            x = test_x[i:i + b_size]
            y = test_y[i:i + b_size]
            y_pred = model(x[:, [0]], x[:, 1:])[:, 0]
            loss = ((y_pred - y)**2).sum()
            avg_loss += loss.item()
        print("test:", epoch, avg_loss / (i + b_size))



