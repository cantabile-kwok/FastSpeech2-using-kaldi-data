import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = torch.nn.Linear(5, 10)
        self.net2 = torch.nn.Linear(5, 15)

    def forward(self, x, x2):
        print(f"inside, before forward, x shape is {x.shape}")
        print(f"inside, before forward, x2 shape is {x2.shape}")
        y = self.net(x)
        print(f"inside, after net1 forward, y shape is {y.shape}")
        y2 = self.net2(x2)
        print(f"inside, after net2 forward, y2 shape is {y2.shape}")
        return y, y2


if __name__ == '__main__':
    data = torch.randn(size=(10, 5))
    data2 = torch.randn(size=(10, 5))
    inp={
        "x": data,
        "x2": data2
    }
    net = Net()
    device = torch.device('cuda')
    net = torch.nn.DataParallel(net)
    net.to(device)
    data = data.to(device)

    print(f"outside, before forward, data shape is {data.shape}")
    out = net(x=inp['x'], x2=inp['x2'])
    print(f"outsize, after forward, out shape is {out[0].shape}, out2 shape is {out[1].shape}")