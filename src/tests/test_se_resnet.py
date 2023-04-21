import unittest
import torch
from ..models.module_se_resnet import SEResNet


class TestSEResNet(unittest.TestCase):
    def test_forward(self):
        # Create an instance of SEResNet
        output_channels = 30
        blocks = 1
        seresnet = SEResNet(output_channels, blocks)

        # Generate a sample input tensor [128, 128, 75] from MSCN
        batch_size = 128
        input_channels = 192
        seq_len = 75
        sample_input = torch.randn(batch_size, input_channels, seq_len)

        # Print input dimension size
        print("Input dimension size:", sample_input.shape)

        # Forward pass
        output = seresnet(sample_input)

        # Print output dimension size
        print("Output dimension size:", output.shape)

        # Check if the output dimension size is as expected
        self.assertEqual(output.size(0), batch_size)
        self.assertEqual(output.size(1), output_channels)
        self.assertEqual(output.size(2), seq_len)


if __name__ == "__main__":
    unittest.main()
