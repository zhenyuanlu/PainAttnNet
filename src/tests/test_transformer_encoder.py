import unittest
import torch
from ..models.module_transformer_encoder import EncoderWrapper


class TestEncoderWrapper(unittest.TestCase):
    def test_forward(self):
        # Set parameters
        num_heads = 5
        model_dim = 75
        senet_reduced_size = 30
        d_mlp = 120
        dropout = 0.1
        N = 2

        # Create an instance of EncoderWrapper
        encoder = EncoderWrapper(num_heads, model_dim, senet_reduced_size, d_mlp, dropout, N)

        # Generate a sample input tensor
        batch_size = 128
        input_channels = 30
        seq_len = 75
        sample_input = torch.randn(batch_size, input_channels, seq_len)

        # Print input dimension size
        print("Input dimension size:", sample_input.shape)

        # Forward pass
        output = encoder(sample_input)

        # Print output dimension size
        print("Output dimension size:", output.shape)

        # Check if the output dimension size is as expected
        self.assertEqual(output.size(0), batch_size)
        self.assertEqual(output.size(1), senet_reduced_size)
        self.assertEqual(output.size(2), seq_len)

if __name__ == "__main__":
    unittest.main()
