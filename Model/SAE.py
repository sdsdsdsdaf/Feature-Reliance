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

    def encode_pre(self, x):
        """ReLU 이전의 pre-activation.

        ghost grad가 죽은 latent에 ReLU 대신 exp()를 태우려면 이 값이 필요하다 —
        ReLU를 통과한 뒤에는 음수 pre-activation이 전부 0으로 뭉개져 기울기가 사라진다."""
        return F.linear(x - self.b_dec, self.W_enc.T, self.b_enc)

    def encode(self, x):
        return F.relu(self.encode_pre(x))

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

    def forward(self, x, return_pre=False):
        """return_pre=True면 (x_hat, z, hidden_pre)를 돌려준다.

        기본 반환 형태를 (x_hat, z)로 유지하는 이유: 평가·진단 경로 6곳이 전부 2-튜플로
        받고 있고, ghost grad가 필요한 곳은 학습 스텝 하나뿐이다."""
        hidden_pre = self.encode_pre(x)
        z = F.relu(hidden_pre)
        x_hat = self.decode(z)
        return (x_hat, z, hidden_pre) if return_pre else (x_hat, z)
