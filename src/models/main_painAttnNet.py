"""
main_painAttnNet.py

PainAttnNet model: The main model of the paper
Modules: MSCN, SEResNet, and Transformer Encoder
"""
import torch.nn as nn
from .module_mscn import MSCN
from .module_se_resnet import SEResNet
from .module_transformer_encoder import EncoderWrapper


class PainAttnNet(nn.Module):
    """
    PainAttnNet model
    """
    def __init__(self):
        super(PainAttnNet, self).__init__()

        # Number of Transformer Encoder Stacks
        N = 2
        # Model dimension from MSCN
        model_dim = 75
        # Dimension of MLP
        d_mlp = 120
        # Number of attention heads
        num_heads = 5
        dropout = 0.1
        num_classes = 2
        # Output SEResNet size
        senet_reduced_size = 30

        # Multiscale Convolutional Network
        self.mscn = MSCN()
        # SEResNet
        self.seresnet = SEResNet(senet_reduced_size, 1)
        # Transformer Encoder
        self.encoderWrapper = EncoderWrapper(num_heads, model_dim, senet_reduced_size, d_mlp, dropout, N)
        # Fully connected layer to output the final prediction
        self.fc = nn.Linear(model_dim * senet_reduced_size, num_classes)

    def forward(self, x):
        mscn_feat = self.mscn(x)
        se_feat = self.seresnet(mscn_feat)
        transformer_feat = self.encoderWrapper(se_feat)
        # Flatten the output of Transformer Encoder to feed into the fully connected layer
        transformer_feat = transformer_feat.contiguous().view(transformer_feat.shape[0], -1)
        final_output = self.fc(transformer_feat)
        return final_output
