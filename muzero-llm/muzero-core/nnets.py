from torch import nn

class ResidualLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.5, use_residual=True, use_norm=True):
        super(ResidualLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )
        self.use_residual = use_residual
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output = self.layer(input)
        if self.use_residual:
            output = output + input
        if self.use_norm:
            output = self.norm(output)
        return output