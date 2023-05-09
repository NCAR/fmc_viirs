from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torch, logging


def init_weights(net, init_type='normal', init_gain=0.0, verbose=True):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if verbose:
        logging.info('Initializing network with %s' % init_type)
    net.apply(init_func)
    
    
def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


class DNN(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize, block_sizes = [1000], dr = [0.5]):
        super(DNN, self).__init__()
        
        if len(block_sizes) > 0:
            blocks = self.block(inputSize, block_sizes[0], dr[0])
            if len(block_sizes) > 1:
                for i in range(len(block_sizes)-1):
                    blocks += self.block(block_sizes[i], block_sizes[i+1], dr[i])
            blocks.append(torch.nn.Linear(block_sizes[-1], outputSize))
        else:
            blocks = [torch.nn.Linear(inputSize, outputSize)]
        
        self.fcn = torch.nn.Sequential(*blocks)
        
    def block(self, inputSize, outputSize, dr):
        return [
            torch.nn.Linear(inputSize, outputSize),
            torch.nn.BatchNorm1d(outputSize),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dr)
        ]

    def forward(self, x):
        x = self.fcn(x)
        return x
    
    def predict(self, x, batch_size = 1024):
        
        if len(x.shape) != 2:
            print(
                f"The input size should be (batch_size, input size), but recieved {x.shape}"
            )
            raise
            
        device = get_device()
        #logger.info(f"Mounting the model to device {device}")
        self.to(device) 
        self.eval()
        
        with torch.no_grad():
            if batch_size > x.shape[0]:
                X = np.array_split(x, x.shape[0] / batch_size)
                pred = torch.cat([
                    self.forward(torch.from_numpy(_x).float().to(device))
                    for _x in X
                ]).cpu()
            else:
                pred = self.forward(
                    torch.from_numpy(x).float().to(device)
                ).cpu()
                
        return pred.numpy()