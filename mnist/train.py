import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class CNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

        # Create architecture
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_features,
                                kernel_size=5)
        self.conv_2 = nn.Conv2d(in_channels=n_features,
                                out_channels=n_features,
                                kernel_size=5)
        self.fc_1 = nn.Linear(n_features * 4 * 4, 50)
        self.fc_2 = nn.Linear(50, 10)

    def forward(self, input_data, verbose=False):
        x = self.conv_1(input_data)
        x = nn.ReLU(x)
        x = nn.MaxPool2d(x, kernel_size=2)

        x = self.conv_2(x)
        x = nn.ReLU(x)
        x = nn.MaxPool2d(x, kernel_size=2)

        x.view(-1, self.n_features * 4 * 4)
        x = self.fc_1(x)
        x = nn.ReLU(x)
        x = nn.LogSoftmax(x, dim=1)

        return x


def train(num_epochs, cnn, train_loader, loss_func, optimizer):
    cnn.train()
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # get batches, normalized
            b_x = Variable(images)
            b_y = Variable(labels)

            output = cnn(b_x)[0]

            loss = loss_func(output, b_y)
            optimizer.zero_grad()  # clear grads for this training step

            # backpropagation, compute gradients
            loss.backward()

            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}],',
                      f'Step [{i + 1}/{total_step}],',
                      f'Loss: {loss.item():.4f}')

    return


if __name__ == '__main__':
    learning_rate = 0.01
    batch_size = 100
    epochs = 10

    # set device to be used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define datasets
    train_data = datasets.MNIST(
        root='data', train=True, transform=ToTensor(), download=True
    )
    test_data = datasets.MNIST(
        root='data', train=False, transform=ToTensor()
    )

    # plot some samples
    fig = plt.figure(figsize=(22, 16))
    cols, rows = (5, 5)

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]

        fig.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()

    # set minibathces, shuffle data, etc
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=True, num_workers=0
    )

    # define network stuff
    model = CNN(24)
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(epochs, model, train_loader, criterion, optimizer)
