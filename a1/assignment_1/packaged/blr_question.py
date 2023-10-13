import numpy as np

################################################################
##### BLR Question Code
################################################################

def single_EM_iter_blr(features, targets, alpha_i, beta_i):
    # Given the old alpha_i and beta_i, computes expectation of latent variable w: M_n and S_n,
    # and using that computes the new alpha and beta values.
    # Should return M_n, S_n, new_alpha, new_beta in that order, with return shapes (M,1), (M,M), None, None
    ### CODE HERE ###
    N, M      = features.shape
    sn        = np.linalg.inv(alpha_i * np.eye(M) + beta_i * features.T @ features) # S_N ∈(M,M)
    mn        = beta_i * sn @ features.T @ targets                                  # M_n ∈(M,1)
    new_alpha = M / (mn.T @ mn + np.trace(sn))                                      # new alpha is a float
    new_beta  = N / ((np.linalg.norm(targets - features @ mn)) ** 2 + np.trace(features.T @ features @ sn)) # new beta is a float
    
    return mn, sn, new_alpha[0][0], new_beta # (M,1), (M,M), None, None