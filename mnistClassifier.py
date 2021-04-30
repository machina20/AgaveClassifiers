import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="An MNIST classifier")

    parser.add_argument('-lr', '--learning_rate', help="Learning rate", default='0.01', type=float)
    parser.add_argument('-e', '--epochs', help="Number of Epochs", default='2', type=int)
    args = parser.parse_args()

    lr = args.learning_rate
    EPOCHS = args.epochs




    train = datasets.MNIST('', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ]))

    test = datasets.MNIST('', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return F.log_softmax(x, dim=1)


    net = Net()
    print(net)

    import torch.optim as optim

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr)

    for epoch in range(EPOCHS):  # 3 full passes over the data
        for data in trainset:  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(X.view(-1, 784))  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = F.nll_loss(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            output = net(X.view(-1,784))
            #print(output)
            for idx, i in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))

    cwd = os.getcwd()
    folder = "MNIST_runs"
    path = os.path.join(cwd, folder)
    if not os.path.exists(path):
        os.mkdir(path)

    lr = str(lr)
    EPOCHS = str(EPOCHS)
    acc = str(round(correct/total, 3))
    loss = str(loss)

    filename = "lr" + lr + "_epochs" + EPOCHS + ".txt" 
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write("Learning-rate = " + lr +
                    "\n\nEpochs = " + EPOCHS +
                    "\n\nLoss = " + loss +
                    "\n\nAccuracy = " + acc)

    PATH = './MNIST_runs/mnist_' + lr + '_' + EPOCHS + '.pth'
    torch.save(net.state_dict(), PATH)