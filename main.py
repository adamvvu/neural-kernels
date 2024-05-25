from plotting import *
from models import *
import numpy as np
import os
import tensorflow as tf
import gc
np.random.seed(42)
tf.random.set_seed(42)

###
# Configuration
###

# Results directory
RESULTSDIR = './Results'

dgps = ['DGP1', 'DGP2', 'DGP3']
experiments = []
for dgp in dgps:
    experiments.extend([f'{dgp}{x}' for x in ['a','b','c','d']])
if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)
for exp in experiments:
    exp_path = f'{RESULTSDIR}/{exp}'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

# Global DGP Settings
N = 5000

# Global Model Settings
EPOCHS = 2000
lambda_ = 0.001
BATCHSIZE = 1000
ARCHITECTURE = [32, 32]
DROPOUT = 0.2
latentDim = 100

###
# Synthetic DGPs
###

def DGP1(N=1000, numFeatures=1, sigma=0.1):
    """
    Smooth and cyclical DGP
    """
    # Features
    if numFeatures == 1:
        X = np.random.normal(size=(N, numFeatures))
    else:
        Sigma = np.random.uniform(-1, 1, size=(numFeatures, numFeatures))
        Sigma = Sigma.T @ Sigma
        mu = [0 for _ in range(numFeatures)]
        X = np.random.multivariate_normal(mu, Sigma, size=N)
    epsilon = np.random.normal(0, sigma, size=N)
    Xt = np.linalg.norm(X, axis=-1)
    true_phi = 1 + np.abs(Xt) + 2*np.sin(Xt**2) + 3*np.cos(Xt**3)
    Y = true_phi + epsilon*Xt
    return X, Y, true_phi

def DGP2(N=1000, numFeatures=1, sigma=0.1):
    """
    Smooth peak DGP
    """
    # Features
    if numFeatures == 1:
        X = np.random.normal(size=(N, numFeatures))
    else:
        Sigma = np.random.uniform(-1, 1, size=(numFeatures, numFeatures))
        Sigma = Sigma.T @ Sigma
        mu = [0 for _ in range(numFeatures)]
        X = np.random.multivariate_normal(mu, Sigma, size=N)
    epsilon = np.random.normal(0, sigma, size=N)
    Xt = np.linalg.norm(X, axis=-1)
    true_phi = np.arctan(Xt)
    Y = true_phi + epsilon*np.linalg.norm(X, axis=-1)
    return X, Y, true_phi

def DGP3(N=1000, sigma=0.1):
    """
    Discontinuous univariate DGP
    """
    X = np.random.normal(size=N)
    epsilon = np.random.normal(0, sigma, size=N)
    mask = np.array(~np.isclose(X, 0), dtype=int)
    true_phi = np.zeros(N)
    for n in range(N):
        if mask[n] != 0:
            x = 1 / X[n]
            true_phi[n] = np.sign(x) * min(abs(x), 10)
    X = np.expand_dims(X, axis=-1)
    Y = true_phi + epsilon*np.linalg.norm(X, axis=-1)*mask
    return X, Y, true_phi

###
# Main Experiment Code
###

def RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP1', plot_mean=True):
    """
    Runs the experiment for a given DGP.
    """
    print(f'Running Experiment for {exp_name}...')

    numFeatures = X.shape[-1]

    # Benchmark: Kernel Ridge Regression
    Y_hat_K, Y_hat_test_K = KernelRidgeRegression(X, Y, X_test, Y_test, lambda_=lambda_, kernel='rbf')

    # Benchmark: DNN
    Y_hat_DNN, Y_hat_test_DNN = StandardDNN(X, Y, X_test, Y_test, 
                                            architecture=ARCHITECTURE, 
                                            dropout=DROPOUT, 
                                            epochs=EPOCHS, 
                                            batch_size=BATCHSIZE,
                                            lambda_=lambda_,
                                            verbose=0)

    # Neural Kernels
    model = NeuralKernelModel(numFeatures, 
                            architecture=ARCHITECTURE, 
                            dropout=DROPOUT,
                            latentDim=latentDim,
                            lambda_=lambda_
                            )
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCHSIZE, verbose=0)
    Y_hat_NK, Y_hat_test_NK = model.predict(X), model.predict(X_test)

    results = {
        'Kernel' : (Y_hat_K, Y_hat_test_K),
        'DNN' : (Y_hat_DNN, Y_hat_test_DNN),
        'Neural Kernel' : (Y_hat_NK, Y_hat_test_NK)
    }

    # Evaluate metrics
    with open(f'{RESULTSDIR}/{exp_name}/metrics.txt', 'w') as file:
        for name, res in results.items():
            y_hat, y_hat_test = res

            # MSE
            mse = np.mean(np.square(Y - y_hat))
            mse_test = np.mean(np.square(Y_test - y_hat_test))
            file.write(f'[{name}] \n')
            file.write(f'MSE: {mse} | MSE (test): {mse_test} \n')

    # Plots
    path = f'{RESULTSDIR}/{exp_name}'
    y_hats = [Y_hat_K, Y_hat_DNN, Y_hat_NK]
    y_hats_test = [Y_hat_test_K, Y_hat_test_DNN, Y_hat_test_NK]
    PlotResiduals(Y, Y_test, y_hats, y_hats_test, path=f'{path}/Residuals.png', title=f'{exp_name}')
    if plot_mean:
        PlotCondMean(X, Y, phi, X_test, Y_test, phi_test, y_hats, y_hats_test, path=f'{path}/CondMean.png', title=f'{exp_name}')

    # Clean up
    del model
    gc.collect()

    return

###
# Experiment 1a: Smooth Cyclical Regression (Low Noise)
###

# DGP Settings
numFeatures = 1
sigma = 0.1

X, Y, phi = DGP1(N=N, numFeatures=numFeatures, sigma=sigma)
X_test, Y_test, phi_test = DGP1(N=N, numFeatures=numFeatures, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP1a')


###
# Experiment 1b: Smooth Cyclical Regression (High Noise)
###

# DGP Settings
numFeatures = 1
sigma = 2

X, Y, phi = DGP1(N=N, numFeatures=numFeatures, sigma=sigma)
X_test, Y_test, phi_test = DGP1(N=N, numFeatures=numFeatures, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP1b')

###
# Experiment 1c: Smooth Cyclical Regression (High Dimensional; Low Noise)
###

# DGP Settings
numFeatures = 50
sigma = 0.1

X, Y, phi = DGP1(N=N, numFeatures=numFeatures, sigma=sigma)
X_test, Y_test, phi_test = DGP1(N=N, numFeatures=numFeatures, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP1c', plot_mean=False)

###
# Experiment 1d: Smooth Cyclical Regression (High Dimensional; High Noise)
###

# DGP Settings
numFeatures = 50
sigma = 2

X, Y, phi = DGP1(N=N, numFeatures=numFeatures, sigma=sigma)
X_test, Y_test, phi_test = DGP1(N=N, numFeatures=numFeatures, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP1d', plot_mean=False)


###
# Experiment 2a: Sharp Peak Regression (Low Noise)
###

# DGP Settings
numFeatures = 1
sigma = 0.1

X, Y, phi = DGP2(N=N, numFeatures=numFeatures, sigma=sigma)
X_test, Y_test, phi_test = DGP2(N=N, numFeatures=numFeatures, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP2a')

###
# Experiment 2b: Sharp Peak Regression (High Noise)
###

# DGP Settings
numFeatures = 1
sigma = 2

X, Y, phi = DGP2(N=N, numFeatures=numFeatures, sigma=sigma)
X_test, Y_test, phi_test = DGP2(N=N, numFeatures=numFeatures, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP2b')

###
# Experiment 2c: Sharp Peak Regression (High Dimensional; Low Noise)
###

# DGP Settings
numFeatures = 50
sigma = 0.1

X, Y, phi = DGP2(N=N, numFeatures=numFeatures, sigma=sigma)
X_test, Y_test, phi_test = DGP2(N=N, numFeatures=numFeatures, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP2c', plot_mean=False)

###
# Experiment 2d: Sharp Peak Regression (High Dimensional; High Noise)
###

# DGP Settings
numFeatures = 50
sigma = 2

X, Y, phi = DGP2(N=N, numFeatures=numFeatures, sigma=sigma)
X_test, Y_test, phi_test = DGP2(N=N, numFeatures=numFeatures, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP2d', plot_mean=False)

###
# Experiment 3a: Discontinuous Regression (Low Noise)
###

# DGP Settings
sigma = 0.1

X, Y, phi = DGP3(N=N, sigma=sigma)
X_test, Y_test, phi_test = DGP3(N=N, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP3a')


###
# Experiment 3b: Discontinuous Regression (High Noise)
###

# DGP Settings
sigma = 2

X, Y, phi = DGP3(N=N, sigma=sigma)
X_test, Y_test, phi_test = DGP3(N=N, sigma=sigma)

RunExperiment(X, Y, phi, X_test, Y_test, phi_test, exp_name='DGP3b')
