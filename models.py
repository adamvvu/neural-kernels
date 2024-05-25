from sklearn.kernel_ridge import KernelRidge
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tqdm import tqdm
import sys

def KernelRidgeRegression(X, Y, X_test, Y_test, lambda_=0., kernel='rbf'):
    """
    Runs a Kernel ridge regression with a pre-specified parametric kernel.
    """

    # Training
    krr = KernelRidge(alpha=lambda_, kernel=kernel)
    krr.fit(X, Y)

    # Estimates
    Y_hat = krr.predict(X)
    Y_hat_test = krr.predict(X_test)

    return Y_hat, Y_hat_test

def StandardDNN(X, Y, X_test, Y_test, architecture=[32,32], dropout=0.2, epochs=100, batch_size=100, lambda_=0., verbose=0):
    """
    Runs a standard DNN for nonparametric regression.
    """
    numFeatures = X.shape[-1]

    # Compile DNN
    inputLayer = tfk.Input(shape=(numFeatures,))
    x = tfk.layers.Dense(architecture[0], 
                         activation='relu',
                         kernel_regularizer=tfk.regularizers.L2(lambda_))(inputLayer)
    for node in architecture[1:]:
        x = tfk.layers.Dropout(dropout)(x)
        x = tfk.layers.Dense(node, 
                             activation='relu',
                             kernel_regularizer=tfk.regularizers.L2(lambda_))(x)
    x = tfk.layers.Dense(1)(x)
    model = tfk.Model(inputLayer, x)
    model.compile(optimizer='adam', loss='mse')

    # Training
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Estimates
    Y_hat = np.ravel(model.predict(X))
    Y_hat_test = np.ravel(model.predict(X_test))

    return Y_hat, Y_hat_test

class NeuralKernelModel:

    def __init__(self, numFeatures, architecture=[32,32], dropout=0.2, latentDim=None, lambda_=0., optimizeAlpha=True, DTYPE=tf.float32):
        """
        Initializes the model for neural kernels.
        
        Args:  numFeatures    (int)   Dimensionality of X
               architecture   (list)  Hidden layers of DNN
               dropout        (float) Dropout rate
               latentDim      (int)   Dimension of latent representation of weights. Defaults to N
               lambda_        (float) Non-negative regularization
               optimizeAlpha  (bool)  Whether to train alpha and DNN jointly
               DTYPE          (obj)   Floating point precision. Defaults to `tf.float32`
        """
        self.numFeatures = numFeatures
        self.architecture = architecture
        self.dropout = dropout
        self.latentDim = latentDim
        self.lambda_ = lambda_
        self.optimizeAlpha = optimizeAlpha
        self.DTYPE = DTYPE
        
        self.compiled = False
        self.alpha = None
        self.model = None
        
        self.losses = []

        assert lambda_ >= 0
    
    def _compile(self):
        """
        Initializes and compiles the DNN.
        """
        inputLayer = tfk.Input(shape=(self.numFeatures,))
        x = tfk.layers.Dense(self.architecture[0], activation='relu')(inputLayer)
        for node in self.architecture[1:]:
            x = tfk.layers.Dropout(self.dropout)(x)
            x = tfk.layers.Dense(node, activation='relu')(x)
        weightLayer = tfk.layers.Dense(self.latentDim, activation='softmax')(x)
        self.model = tfk.Model(inputLayer, weightLayer)
        
        return
    
    def loss(self, x_batch, y_batch, step_idx):
        """
        Computes the L2 loss function.
        """
        K = self.construct_kernel(x_batch, training=True)
        y_hat_batch = tf.linalg.matmul(K, self.alpha)
        
        # MSE
        lossval = tf.math.reduce_mean(tf.math.square(y_batch - y_hat_batch))
        
        # Regularization
        reg = 0
        if self.lambda_ > 0:
            reg = self.lambda_ * tf.linalg.matmul(
                                    tf.linalg.matmul(self.alpha[step_idx[0]:step_idx[1],:], K, transpose_a=True), 
                                    self.alpha)
        
        return lossval + reg
    
    def construct_kernel(self, X, training=False):
        """
        Constructs the kernel matrix.
        """
        W = self.model(self.X, training=training)
        Wx = self.model(X, training=training)
        K = tf.linalg.matmul(Wx, W, transpose_b=True)
        return K
    
    def predict(self, X):
        """
        Makes predictions.
        """
        K = self.construct_kernel(X)
        Y_hat = tf.linalg.matmul(K, self.alpha)
        return np.ravel(Y_hat.numpy())
    
    def fit(self, X, Y, epochs=10, batch_size=64, verbose=0):
        """
        Fits the model.
        """

        # Initialize alpha
        N = Y.shape[0]
        self.alpha = np.expand_dims(np.ones(N)*np.mean(Y)*1/N, axis=-1)
        if self.optimizeAlpha:
            self.alpha = tf.Variable(self.alpha, dtype=self.DTYPE)
        else:
            self.alpha = tf.constant(self.alpha, dtype=self.DTYPE)

        # Initialize DNN
        if self.latentDim is None:
            self.latentDim = N
        if not self.compiled:
            self._compile()
        
        # Process data
        Y = tf.constant(np.expand_dims(Y, axis=-1), dtype=self.DTYPE)
        X = tf.constant(X, dtype=self.DTYPE)
        self.X = X
        self.Y = Y
        self.N = N
        
        df_train = tf.data.Dataset.from_tensor_slices((X, Y))
        df_train = df_train.shuffle(buffer_size=1024).batch(batch_size)
        
        # Training loop
        optimizer = tfk.optimizers.Adam()
        for epoch in tqdm(range(epochs), file=sys.stdout):
            for step, (x_batch_train, y_batch_train) in enumerate(df_train):
                step_idx = (step*batch_size, (step+1)*batch_size)
                
                # Optimal alpha (For L2 penalty)
                if not self.optimizeAlpha:
                    self.alpha = self.optimal_alpha()
                
                # Compute gradients
                with tf.GradientTape() as tape:
                    loss = self.loss(x_batch_train, y_batch_train, step_idx)
                    
                # Backpropagate
                params = self.model.trainable_weights
                if self.optimizeAlpha:
                    params = params + [self.alpha]
                grads = tape.gradient(loss, params)

                # Gradient step
                optimizer.apply_gradients(zip(grads, params))
            loss = float(np.ravel(loss.numpy())[0])
            if verbose!=0:
                tqdm.write(f'Epoch: {epoch} | Loss: {loss}')
            self.losses.append(float(loss))
        
        # Set globally optimal alpha post-training
        self.alpha = self.optimal_alpha()
        
        return self.losses
    
    def optimal_alpha(self):
        """
        Computes the globally optimal alpha
        """
        K = self.construct_kernel(self.X)
        eps = 1e-4
        alpha = tf.linalg.matmul(tf.linalg.inv(K + self.N*(self.lambda_+eps)*tf.eye(self.N)), self.Y)
        return alpha