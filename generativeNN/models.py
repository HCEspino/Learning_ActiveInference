import torch.nn as nn

class TransitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, state_size=4):
        super(TransitionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = state_size

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activ1 = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activ2 = nn.LeakyReLU(0.1)

        #Split into two
        self.mean_layer = nn.Linear(hidden_size, state_size)
        self.var_layer = nn.Linear(hidden_size, state_size)
        self.var_activ = nn.Softplus()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)


        mean = self.mean_layer(x)
        var = self.var_layer(x)
        var = self.var_activ(var)
        return mean, var
    
class PosteriorModel(nn.Module):
    def __init__(self, input_size, hidden_size, state_size=4):
        super(PosteriorModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = state_size

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activ1 = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activ2 = nn.LeakyReLU(0.1)

        #Split into two
        self.mean_layer = nn.Linear(hidden_size, state_size)
        self.var_layer = nn.Linear(hidden_size, state_size)
        self.var_activ = nn.Softplus()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)


        mean = self.mean_layer(x)
        var = self.var_layer(x)
        var = self.var_activ(var)
        return mean, var
    
class LikelihoodModel(nn.Module):
    def __init__(self, input_size, hidden_size, state_size=4):
        super(LikelihoodModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = state_size

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activ1 = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activ2 = nn.LeakyReLU(0.1)

        #Split into two
        self.mean_layer = nn.Linear(hidden_size, state_size)
        self.var_layer = nn.Linear(hidden_size, state_size)
        self.var_activ = nn.Softplus()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)


        mean = self.mean_layer(x)
        var = self.var_layer(x)
        var = self.var_activ(var)
        return mean, var