import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import chi2
from scipy.stats import norm
from matplotlib.patches import Ellipse
import matplotlib.colors as mcol

################################################################
##### EMM Framework Code
################################################################

# plot_cov_ellipse adapted from:
# http://www.nhsilbert.net/source/2014/06/bivariate-normal-ellipse-plotting-in-python/

def plot_GMM_components(mu, Sigma, colours, Ns, *args, **kwargs):
    '''
    Plot ellipses for the bivariate normals with mean mu[:,i] and covariance Sigma[:,:,i]
    '''
    x_vals = np.linspace(-3, 3, 1_000)

    #assert mu.shape[1] == Sigma.shape[1]
    for i in range(mu.shape[0]):
        kwargs['c'] = colours[i]
        ys = norm.pdf(x_vals, mu[i], Sigma[i])
        plt.plot(x_vals, ys, *args, **kwargs)

def plot_GMM_data(gaussian_1, gaussian_2, gaussian_3, colors=['b','r','g']):
    x_max = max(np.max(gaussian_1), np.max(gaussian_2), np.max(gaussian_3))
    x_min = min(np.min(gaussian_1), np.min(gaussian_2), np.min(gaussian_3))

    x_min_limit = x_min
    x_max_limit = x_max
    
    bins = np.linspace(x_min_limit, x_max_limit, 100)

    plt.figure(figsize=(8,8))
    plt.hist(gaussian_1, bins, alpha=0.5, color=colors[0], density=True)
    plt.hist(gaussian_2, bins, alpha=0.5, color=colors[1], density=True)
    plt.hist(gaussian_3, bins, alpha=0.5, color=colors[2], density=True)
    plt.xlim([-3, 3])

def make_GMM_data():
    np.random.seed(0)
    Nk_gt = np.array([200,160,400]).reshape(-1,1)
    pi_gt = Nk_gt / Nk_gt.sum()
    mu_gt = np.stack([np.array([0.5]), np.array([-1.3]), np.array([2])], axis=0)
    std_gt = np.stack([(np.array([0.4])), (np.array([0.6])), (np.array([0.2]))], axis=0)
    gaussian_1 = np.random.randn(200) * std_gt[0] + mu_gt[0]
    gaussian_2 = np.random.randn(160) * std_gt[1] + mu_gt[1]
    gaussian_3 = np.random.randn(400) * std_gt[2] + mu_gt[2]
    data = np.concatenate([gaussian_1, gaussian_2, gaussian_3], axis=0).squeeze()
    return pi_gt, mu_gt, std_gt, gaussian_1, gaussian_2, gaussian_3, data, Nk_gt

def gmm_to_eta(mu, var):
    return np.array([
        mu / var,
        -0.5 * (1 / var)
    ])

def gmm_from_eta(eta):
    return np.array([
        - 0.5 * (eta[0] / eta[1]),
        - 0.5 * (1 / eta[1])
    ])

def gmm_exp_to_nat(l):
    return np.array([
        l[0] / (l[1] - l[0] * l[0]),
        -0.5 * 1 / (l[1] - l[0] * l[0])
    ])

def gmm_sufstat(data):
    if isinstance(data, np.ndarray) and len(data) > 1:
        return np.stack([data, data*data], axis=1)
    else:
        return np.stack([data, data*data])

################################################################
##### BLR Framework Code
################################################################

def make_blr_data():
    np.random.seed(0)
    # Set Up Data for the Straight Line problem
    true_w = np.array([-0.3, 0.5]) # (2,)
    true_function = lambda x: true_w[0] + true_w[1]*x
    line_x = np.linspace(-1,1,100) # (100,)
    line_y = true_function(line_x) # (100,)
    num_points = 30 # We will refer to this as n
    noise_sigma = 0.2
    beta = 1 / (noise_sigma**2)
    data_x = np.random.rand(num_points)*2 -1 # (n,)
    data_y = true_function(data_x) + np.random.randn(num_points)*noise_sigma # (n,)
    return line_x, line_y, data_x, data_y, true_w

def plot_blr_function(ax, line_x, line_y, data_x, data_y, pred_y=None, std_dev=None):
    ax.plot(line_x, line_y, c='r', label='True Function')
    ax.scatter(data_x, data_y, label='Data')
    if pred_y is not None:
        ax.plot(line_x, pred_y.reshape(-1), label='Prediction')
        if std_dev is not None:
            upper_line = pred_y.reshape(-1) + std_dev.reshape(-1)
            lower_line = pred_y.reshape(-1) - std_dev.reshape(-1)
            ax.fill_between(line_x, lower_line, upper_line, alpha=0.4, color='r')
    ax.legend()

def plot_blr_gaussian(ax, mean, cov, true_w):
    x, y = np.mgrid[-1:2:0.1, -1:2:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean.reshape(-1), cov)
    ax.contourf(x, y, rv.pdf(pos))
    ax.set_xlabel('$w_0$'); ax.set_ylabel('$w_1$')
    ax.scatter(true_w[0], true_w[1], c='r', label='True $\mathbf{w}$')
    ax.legend()

def run_simple_blr_model(single_EM_iter_blr, initial_alpha, initial_beta, line_x, line_y, data_x, data_y, features, targets, true_w):
    if initial_alpha is None or initial_beta is None:
        print("value for alpha or beta is None")
        return
    iters = 10
    fig, axs = plt.subplots(iters, 2, figsize=(14,iters*5))
    alpha_i, beta_i = initial_alpha, initial_beta
    for i in range(iters):
        new_w_mean, new_w_cov, alpha_i, beta_i = single_EM_iter_blr(features, targets, alpha_i, beta_i)
        #print("iter %.3f, %.3f, %.3f" % (i+1, alpha_i, beta_i))
        print("iter {}, alpha={:.3f}, beta={:.3f}".format(i+1,alpha_i, beta_i))
        if new_w_mean is None or new_w_cov is None:
            print("single_EM_iter has not been implemented")
            return
        pred_y = make_phi(line_x) @ new_w_mean
        plot_blr_function(axs[i,0], line_x, line_y, data_x, data_y, pred_y=pred_y)
        axs[i,0].set_title('Updated prediction after {} iterations'.format(i+1))
        plot_blr_gaussian(axs[i,1],new_w_mean, new_w_cov, true_w)
        axs[i,1].set_title('Posterior over $\mathbf{{w}}$ after {} iterations'.format(i+1))

def make_phi(data):
    # Takes 1-D data and maps it into d=2 dimensional features
    data_vec = data.reshape(-1) # (n,)
    return np.stack([np.ones_like(data_vec), data_vec], axis=0).T # (n,d)