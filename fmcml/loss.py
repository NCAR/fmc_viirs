import torch, logging

def load_loss(loss_type):
    logging.info(f"Loading loss {loss_type}")
    if loss_type == "mse":
        return torch.nn.MSELoss()
    if loss_type == "mae":
        return torch.nn.L1Loss()
    if loss_type == "huber":
        return torch.nn.SmoothL1Loss()
    if loss_type == "logcosh":
        return LogCoshLoss()
    if loss_type == "xtanh":
        return XTanhLoss()
    if loss_type == "xsigmoid":
        return XSigmoidLoss()
    

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-12
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-12
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t )