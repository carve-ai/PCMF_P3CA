import numpy as np
cimport numpy as np
cimport cython

## Compile-time datatypes
DTYPE_float = np.float64
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int32
ctypedef np.int_t DTYPE_int_t

@cython.boundscheck(False)
@cython.wraparound(False) 
def group_soft_threshold(np.ndarray[DTYPE_float_t, ndim=1] vec, alpha):
    """
    Apply ℓ2 (group) soft-thresholding to a single vector, in place.

    This computes:
        vec ← (1 - alpha / ||vec||_2)_+ * vec
    where (·)_+ = max(·, 0). If ||vec||_2 ≤ alpha, the vector is set to 0.

    Args:
        vec (ndarray[float64], shape (p,)):
            Input vector to be thresholded. Modified in place.
        alpha (float):
            Nonnegative threshold parameter.

    Returns:
        ndarray[float64]:
            The same array `vec` after in-place thresholding.

    Notes:
        • Expects finite `alpha` and finite entries in `vec`.
        • No defensive copies are made; callers must copy beforehand if needed.
    """
    cdef int n = vec.shape[0]
    cdef double vec_norm = 0.0

    for i in range(n):
        vec_norm += vec[i]*vec[i]
    vec_norm = np.sqrt(vec_norm)
    if vec_norm > alpha:
        for i in range(n):
            vec[i] = vec[i] - alpha * vec[i] / vec_norm
        return vec
    else:
        for i in range(n):
            vec[i] = 0.0
        return vec

@cython.boundscheck(False)
@cython.wraparound(False) 
def prox_c(np.ndarray[DTYPE_float_t, ndim=2] V, lamb, rho, np.ndarray[DTYPE_float_t, ndim=1] w):
    """
    Row-wise group-lasso proximal operator, in place.

    For each row i of V, applies `group_soft_threshold` with
        alpha_i = w[i] * lamb / rho

    Args:
        V (ndarray[float64], shape (n, p)):
            Matrix whose rows form the groups. Modified in place.
        lamb (float):
            Group-lasso penalty parameter (λ ≥ 0).
        rho (float):
            ADMM (or algorithmic) quadratic penalty parameter (ρ > 0).
        w (ndarray[float64], shape (n,)):
            Nonnegative per-row weights.

    Returns:
        ndarray[float64]:
            The same array `V` after in-place proximal updates.

    Notes:
        • All inputs must be finite; this implementation does not handle `np.inf`.
        • For maximum performance, pass C-contiguous float64 arrays.
        • If you need a non-mutating version, pass `V.copy()` instead.
    """
    n = V.shape[0]
    for i in range(n):
        alpha = w[i]*lamb/rho
        V[i,:] = group_soft_threshold(V[i,:],alpha)
    return V

if __name__=="__main__":
   pass