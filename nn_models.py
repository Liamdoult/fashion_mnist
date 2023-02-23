import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.layer_1_linear = torch.nn.Linear(784, 128)
        self.layer_1_activation = torch.nn.ReLU()

        self.layer_2_linear = torch.nn.Linear(128, 128)
        self.layer_2_activation = torch.nn.ReLU()

        self.layer_3_linear = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer_1_linear(x)
        x = self.layer_1_activation(x)
        x = self.layer_2_linear(x)
        x = self.layer_2_activation(x)
        x = self.layer_3_linear(x)
        return x