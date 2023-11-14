import torch
from torch.linalg import matrix_norm as frob_norm


def orthogonal(Q0, max_iters=30, stop_early=True, tol=1e-6):
    """
    Iterative orthogonalization of columns of a tall matrix.
    
    The output matrix (Q) is differentiable with respect to the input matrix (Q0).

    Args:
        Q0: some random matrix (cannot be degenerate like a matrix of zeros)
        max_iters: maximum number of iterations to run before stopping
        stop_early: whether the loop would terminate if ||Q.T @ Q - I||_F < some tolerance
        tol: used together with stop_early; ignored unless stop_early is true
    
    Returns:
        Q: an orthogonal matrix derived from Q0
        i: number of iterations it tooks for orthogonalization to complete
        
    References:
        Rianne van den Berg et al. Sylvester normalizing flows for variational inference.
        Ake Björck and Clazett Bowie. An iterative algorithm for computing the best estimate of an orthogonal matrix.
    """
    
    assert len(Q0.shape) == 2
    D, M = Q0.shape
    assert D > M
    
    Q = Q0 / frob_norm(Q0)
    I = torch.eye(M)
    
    for i in range(max_iters):
        
        diff = I - Q.T @ Q
        
        if frob_norm(diff) < tol: 
            break
        
        Q = Q @ (I + 0.5 * diff)
        
    # we'd expect convergence to be reached by now
    
    assert frob_norm(I - Q.T @ Q) < tol, "Warning: convergence not reached"
    
    return Q, i