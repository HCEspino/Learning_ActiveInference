import torch.nn as nn

class TransitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=8):
        super(TransitionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activ1 = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activ2 = nn.Tanh()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)
        x = self.layer3(x)
        return x
    
class PosteriorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=8):
        super(PosteriorModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activ1 = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activ2 = nn.Tanh()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)
        x = self.layer3(x)
        return x
    
class LikelihoodModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super(LikelihoodModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activ1 = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activ2 = nn.Tanh()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)
        x = self.layer3(x)
        return x