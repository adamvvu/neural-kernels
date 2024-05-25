import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def PlotResiduals(Y, Y_test, list_of_Y_hat, list_of_Y_hat_test, path, title='', q=0.0):
    """
    Plots the residuals
    """
    idx = [i for i in range(Y.shape[0])]
    idx_test = [i for i in range(Y_test.shape[0])]
    y_max = []
    y_min = []

    fig, ax = plt.subplots(1,2, figsize=(12,7), sharex=True, sharey=True)
    names = ['Kernel', 'DNN', 'Neural Kernel']
    colors = ['blue', 'purple', 'orange']
    for j, (y_hat, y_hat_test) in enumerate(zip(list_of_Y_hat, list_of_Y_hat_test)):
        ax[0].scatter(idx, y_hat-Y, color=colors[j], alpha=0.3, label=names[j])
        ax[1].scatter(idx, y_hat_test-Y_test, color=colors[j], alpha=0.3, label=names[j])
        y_min.append(min(np.quantile(y_hat-Y,q), np.quantile(y_hat_test-Y,q)))
        y_max.append(max(np.quantile(y_hat-Y,1-q), np.quantile(y_hat_test-Y,1-q)))
    y_min = np.min(y_min)
    y_max = np.max(y_max)
    plt.ylim((y_min, y_max))

    ax[0].set_ylabel('Residuals')
    ax[1].set_ylabel('Residuals')
    ax[0].set_title('In-sample')
    ax[1].set_title('Out-of-sample')

    fig.tight_layout()
    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.suptitle(title, x=0.525, y=1)
    plt.savefig(path)
    return 

def PlotCondMean(X, Y, phi, X_test, Y_test, phi_test, list_of_Y_hat, list_of_Y_hat_test, path, title='', q=0.0):
    """
    Plots the conditional mean for univariate X.
    """
    fig, ax = plt.subplots(1,2, figsize=(12,7), sharey=True)
    names = ['Kernel', 'DNN', 'Neural Kernel']
    colors = ['blue', 'purple', 'orange']

    y_max = [max(np.quantile(Y, 1-q), np.quantile(Y_test,1-q))]
    y_min = [min(np.quantile(Y, q), np.quantile(Y,q))]
    
    ax[0].scatter(X, phi, color='red', alpha=0.4)
    ax[0].scatter(X, Y, color='grey', alpha=0.2)
    ax[1].scatter(X_test, phi_test, color='red', alpha=0.4, label='True $\phi$')
    ax[1].scatter(X_test, Y_test, color='grey', alpha=0.2, label='Observed $Y$')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[0].set_title('In-sample')
    ax[1].set_title('Out-of-sample')
    for j, (y_hat, y_hat_test) in enumerate(zip(list_of_Y_hat, list_of_Y_hat_test)):
        ax[0].scatter(X, y_hat, color=colors[j], alpha=0.15)
        ax[1].scatter(X_test, y_hat_test, color=colors[j], alpha=0.15, label=names[j])
        y_min.append(min(np.quantile(y_hat,q), np.quantile(y_hat_test,q)))
        y_max.append(max(np.quantile(y_hat,1-q), np.quantile(y_hat_test,1-q)))
    y_min = np.min(y_min)
    y_max = np.max(y_max)
    plt.ylim((y_min, y_max))

    fig.tight_layout()
    plt.suptitle(title, x=0.525, y=1)
    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.savefig(path)
    return





