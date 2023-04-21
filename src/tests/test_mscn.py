import unittest
import torch
from ..models.module_mscn import MSCN


class TestMSCN(unittest.TestCase):
    def test_forward(self):
        # Create an instance of MSCN
        mscn = MSCN()

        # Input shape for BioVid is torch.Size([128, 1, 2816]) if we have batch size of 128
        batch_size = 128
        seq_len = 2816
        channels = 1
        sample_input = torch.randn(batch_size, channels, seq_len)

        # Print input dimension size
        print("Input dimension size:", sample_input.shape)

        # Forward pass
        output = mscn(sample_input)

        # Print output dimension size
        print("Output dimension size:", output.shape)

        # Check if the output dimension size is as expected
        self.assertEqual(output.size(0), batch_size)
        # Check if the output channel size is as expected
        self.assertEqual(output.size(1), 192)
        # Check if the output length size is as expected
        self.assertEqual(output.size(2), 75)


if __name__ == "__main__":
    unittest.main()
