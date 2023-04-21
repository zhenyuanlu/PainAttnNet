import torch
import torch.nn as nn
from ..models.module_mscn import MSCN
from ..models.module_se_resnet import SEResNet
from ..models.module_transformer_encoder import EncoderWrapper
from ..models.main_painAttnNet import PainAttnNet

# Include the PainAttnNet class here

# Unit test
import unittest


class TestPainAttnNetOutput(unittest.TestCase):
    def test_output_dimensions(self):
        model = PainAttnNet()
        # Input shape for BioVid is torch.Size([128, 1, 2816]) if we have batch size of 128
        batch_size = 128
        input_channels = 1
        seq_len = 2816
        sample_input = torch.randn(batch_size, input_channels, seq_len)

        # Forward pass
        output = model(sample_input)

        # Print output dimension size
        print("PainAttnNet output dimension size:", output.shape)

        # Check if the output dimension size is as expected
        num_classes = 2
        self.assertEqual(output.size(0), batch_size)
        self.assertEqual(output.size(1), num_classes)


if __name__ == "__main__":
    unittest.main()
