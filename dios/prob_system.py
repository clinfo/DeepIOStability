class Encoder(torch.nn.Module):
    def __init__(self, obs_dim, h_dim, state_dim, activation=F.relu):
        super(Encoder, self).__init__()
        # 2 hidden layeros encoder
        self.fc_e0 = nn.Linear(obs_dim, h_dim)
        self.fc_e1 = nn.Linear(h_dim, h_dim)
        self.fc_mean = nn.Linear(h_dim, state_dim)
        self.fc_var = nn.Linear(h_dim, state_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))
        z_mean = self.fc_mean(x)
        z_var = F.softplus(self.fc_var(x))
        return z_mean, z_var


class Decoder(torch.nn.Module):
    def __init__(self, obs_dim, h_dim, state_dim, activation=F.relu):
        super(Decoder, self).__init__()
        # 2 hidden layeros encoder
        self.fc_e0 = nn.Linear(obs_dim, h_dim)
        self.fc_e1 = nn.Linear(h_dim, h_dim)
        self.fc_mean = nn.Linear(h_dim, state_dim)
        self.fc_var = nn.Linear(h_dim, state_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))
        z_mean = self.fc_mean(x)
        z_var = F.softplus(self.fc_var(x))
        return z_mean, z_var
