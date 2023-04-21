"""
test_generate_kfolds_index.py

Test the generate_kfolds_index function in utils.py
"""

import unittest
import os
import numpy as np
from ..utils.utils import generate_kfolds_index
import glob

test_npz_dir = r'E:\research\phd-research-projects\unit_test_data\processed_bioVid_partA_test'


class TestGenerateKFoldsIndex(unittest.TestCase):
    def setUp(self):
        # dir npz files
        self.test_dir = test_npz_dir

        # def setUp(self):
        #     # Create a temporary directory for npz files
        #     self.temp_dir = 'temp_npz_files'
        #     os.makedirs(self.temp_dir)
        # # Create 10 dummy npz files
        # for i in range(10):
        #     np.savez(os.path.join(self.test_dir, f'dummy_{i}.npz'), x=np.random.rand(10, 10), y=np.random.rand(10))

    def test_generate_kfolds_index(self):
        k_folds = 5 # k_folds must >1
        kfolds_index = generate_kfolds_index(self.test_dir, k_folds)

        # Check if the number of folds is equal to k_folds
        self.assertEqual(len(kfolds_index), k_folds)

        npz_files = glob.glob(os.path.join(test_npz_dir, '*.npz'))
        npz_files = np.asarray(npz_files)
        expected_sample_sizes = np.array_split(npz_files, k_folds)

        # Check if each fold has a training and testing set
        for fold_index, (train_data, test_data) in kfolds_index.items():
            self.assertTrue(len(train_data) > 0)
            self.assertTrue(len(test_data) > 0)

            # Check if there's no intersection between training and testing sets
            self.assertTrue(set(train_data).isdisjoint(set(test_data)))

            # Check if all files in the training and testing sets are npz files
            self.assertTrue(all(file.endswith('.npz') for file in train_data))
            self.assertTrue(all(file.endswith('.npz') for file in test_data))

            # Check if the sample size in each fold matches the expected size
            self.assertEqual(len(test_data), len(expected_sample_sizes[fold_index]))


if __name__ == '__main__':
    unittest.main()
