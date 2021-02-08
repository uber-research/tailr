##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'utils.py'
"""

import unittest
import torch
from utils import kl_gaussian, kl_onehotcategorical


class TestUtils(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_kl_gaussian_1(self):
        observed = kl_gaussian(p_mu=torch.tensor([[0.0], [1]]), p_sigma=torch.tensor(
            [[1.0], [1]]), q_mu=torch.tensor([[0.0], [1]]), q_sigma=torch.tensor([[1.0], [1]]))
        expected = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.all(torch.eq(observed, expected)),
                        msg=f'Observed={observed}, Expected={expected}')

    def test_kl_onehotcategorical_1(self):
        observed = kl_onehotcategorical(cluster_score=torch.tensor(
            [[0.5, 0.5]]), prior=torch.tensor([[0.5, 0.5]]))
        expected = torch.tensor([0.0])
        self.assertTrue(torch.all(torch.eq(observed, expected)),
                        msg=f'Obsserved={observed}, Expected={expected}')


if __name__ == '__main__':
    unittest.main()
