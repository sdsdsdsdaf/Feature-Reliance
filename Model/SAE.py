import torch
from torch import nn
import torch.nn.functional as F


class VanillaL1SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, b_dec_init=None, dec_bias_mode="zero"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dec_bias_mode = str(dec_bias_mode).lower()

        if b_dec_init is None:
            if self.dec_bias_mode == "zero":
                b_dec_init = torch.zeros(self.input_dim)
            else:
                raise ValueError("b_dec_init is required for non-zero decoder bias initialization.")
        else:
            b_dec_init = b_dec_init.detach().float()

        self.W_enc = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_dec = nn.Parameter(torch.empty(self.hidden_dim, self.input_dim))
        self.b_dec = nn.Parameter(torch.empty(self.input_dim))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
            self.b_dec.copy_(b_dec_init.reshape(-1).to(dtype=self.b_dec.dtype, device=self.b_dec.device))

    def encode(self, x):
        sae_in = x - self.b_dec
        return F.relu(F.linear(sae_in, self.W_enc.T, self.b_enc))

    def decode(self, z):
        return F.linear(z, self.W_dec.T, self.b_dec)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self, eps=1e-8):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(eps))

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        if self.W_dec.grad is None:
            return
        parallel_component = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
        self.W_dec.grad.sub_(parallel_component * self.W_dec.data)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
