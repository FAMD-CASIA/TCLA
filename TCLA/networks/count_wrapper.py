import torch.nn as nn
import torch.nn.functional as F


class CountWrapper(nn.Module):
    """Wrapper module for converting real valued input/output to rates."""
    def __init__(self, ae_net):
        super().__init__()
        self.ae_net = ae_net

    def forward(self, x, session_id=None):
        if session_id is not None:
            logrates, z, *_ = self.ae_net(x, session_id)
        else:
            logrates, z = self.ae_net(x)
        return F.softplus(logrates), z

    def encode(self, x, session_id=None):
        if session_id is not None:
            return self.ae_net.encode(x, session_id)
        return self.ae_net.encode(x)

    def decode(self, z, session_id=None):
        decoded_output = self.ae_net.decode(z, session_id) if session_id is not None else self.ae_net.decode(z)
        return F.softplus(decoded_output)
