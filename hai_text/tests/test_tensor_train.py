import torch
import unittest
from hai_text.models.tensor_train import TTMatrix, TTTensor

class TestTensorTrain(unittest.TestCase):
    def test_tt_matrix_shape(self):
        # Create a TT Matrix for a 16x16 -> 16x16 transform
        # Decomposed into 2 cores: 4x4 -> 4x4 each.
        in_dims = [4, 4]
        out_dims = [4, 4]
        ranks = [1, 2, 1]
        
        tt_mat = TTMatrix(in_dims, out_dims, ranks)
        
        # Input vector: 16 (4*4)
        x = torch.randn(16)
        # Reshape to (4, 4) for the forward pass if our logic expects structured input
        x_struct = x.reshape(4, 4)
        
        y = tt_mat.metric_projection(x_struct)
        
        self.assertEqual(y.shape, (4, 4))
        
    def test_tt_matrix_batch(self):
        in_dims = [4, 4]
        out_dims = [4, 4]
        ranks = [1, 2, 1]
        tt_mat = TTMatrix(in_dims, out_dims, ranks)
        
        batch_size = 5
        x = torch.randn(batch_size, 4, 4)
        y = tt_mat.metric_projection(x)
        self.assertEqual(y.shape, (batch_size, 4, 4))
        
    def test_tt_tensor_reconstruct(self):
        # Tensor 8x8x8
        dims = [8, 8, 8]
        ranks = [1, 2, 2, 1]
        tt_tens = TTTensor(dims, ranks)
        
        full = tt_tens.reconstruct()
        self.assertEqual(full.shape, (8, 8, 8))
        
    def test_tt_tensor_forward(self):
        # Mocking A-tensor structure: (Vocab, Sem, Syn)
        # e.g., (10, 4, 4)
        dims = [10, 4, 4]
        ranks = [1, 2, 2, 1]
        tt_tens = TTTensor(dims, ranks)
        
        # Test getting the slice for observation 3
        obs_slice = tt_tens.forward(3)
        self.assertEqual(obs_slice.shape, (4, 4))
        
        # Verify it matches reconstruction
        full = tt_tens.reconstruct()
        expected = full[3]
        
        print(f"Slice norm: {obs_slice.norm()}, Expected norm: {expected.norm()}")
        self.assertTrue(torch.allclose(obs_slice, expected, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
