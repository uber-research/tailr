##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'utils.py'
"""

import unittest
import torch
from model import SharedEncoder


class TestUtils(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_shared_encoder_shape(self):
        temp = SharedEncoder(in_dims=1, n_enc=[16, 16], enc_strides=[
                             0, 0], encoder_type='multi')
        res = temp.forward(torch.ones([16, 1]))
        self.assertEqual(res.shape, torch.Size([16, 16]), msg=f"{res.shape}")


if __name__ == '__main__':
    unittest.main()
