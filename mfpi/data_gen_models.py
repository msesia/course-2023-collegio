import numpy as np
from scipy.stats import norm, t

class Model_Reg1:
    def __init__(self, a=0):
        self.a = a
        self.sigma = 0.2

    def sample_X(self, n):
        X = np.random.uniform(0, 1, size=n)
        X = X.reshape((n,1))
        return X.astype(np.float32)

    def sample_Y(self, X):
        x = X[:,0]
        epsilon = self.sigma * np.random.randn(x.shape[0]) 
        Y =  x + epsilon        
        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y
    
    def oracle_predict(self, X, alpha):
        x = X[:,0]
        n = len(x)
        mu = x
        sigma = self.sigma
        lower = norm.ppf(alpha/2, loc=mu, scale=sigma)
        upper = norm.ppf(1.0-alpha/2, loc=mu, scale=sigma)
        return lower, upper


class Model_Reg2:
    def __init__(self, a=0):
        self.a = a
        self.sigma = 0.2

    def sample_X(self, n):
        X = np.random.uniform(0, 1, size=n)
        X = X.reshape((n,1))
        return X.astype(np.float32)

    def sample_Y(self, X):
        x = X[:,0]
        epsilon = self.sigma * (x**2) * np.random.randn(x.shape[0]) 
        Y =  x + epsilon        
        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y
    
    def oracle_predict(self, X, alpha):
        x = X[:,0]
        n = len(x)
        mu = x
        sigma = self.sigma * (x**2)
        lower = norm.ppf(alpha/2, loc=mu, scale=sigma)
        upper = norm.ppf(1.0-alpha/2, loc=mu, scale=sigma)
        return lower, upper

class Model_Reg3:
    def __init__(self, a=0):
        self.a = a
        self.sigma = 0.1
        self.df = 2

    def sample_X(self, n):
        X = np.random.uniform(0, 1, size=n)
        X = X.reshape((n,1))
        return X.astype(np.float32)

    def sample_Y(self, X):
        x = X[:,0]
        epsilon = self.sigma * np.random.standard_t(df=self.df, size=x.shape[0])
        Y =  x + epsilon        
        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y
    
    def oracle_predict(self, X, alpha):
        x = X[:,0]
        n = len(x)
        mu = x
        sigma = self.sigma
        lower = t.ppf(alpha/2, self.df, loc=mu, scale=sigma)
        upper = t.ppf(1.0-alpha/2, self.df, loc=mu, scale=sigma)
        return lower, upper

class Model_Reg4:
    def __init__(self, a=0):
        self.a = a

    def sample_X(self, n):
        X = np.random.uniform(0, 1, size=n)
        X = X.reshape((n,1))
        return X.astype(np.float32)

    def sample_Y(self, X):
        x = X[:,0]
        epsilon = ((1.0-self.a) + 10*self.a*x**2) * np.random.randn(x.shape[0]) 
        Y =  1 * np.sin(4 * np.pi * x) + 0.25 * epsilon        
        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y
    
    def oracle_predict(self, X, alpha):
        x = X[:,0]
        n = len(x)
        mu = 1 * np.sin(4 * np.pi * x)
        sigma = 0.25 * ((1.0-self.a) + 10*self.a*x**2)
        lower = norm.ppf(alpha/2, loc=mu, scale=sigma)
        upper = norm.ppf(1.0-alpha/2, loc=mu, scale=sigma)
        return lower, upper
    
class Model_Reg5:
    def __init__(self, x_shift=0, a=0):
        self.x_shift = x_shift
        self.a = a

    def sample_X(self, n):
        X = self.x_shift + np.abs(np.random.normal(0, 0.5, size=n))
        X = X.reshape((n,1))
        return X.astype(np.float32)

    def sample_Y(self, X):
        x = X[:,0]
        epsilon = ((1.0-self.a) + 10*self.a*x**2) * np.random.randn(x.shape[0]) 
        Y =  1 * np.sin(4 * np.pi * x) + 0.25 * epsilon        
        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y
    
    def oracle_predict(self, X, alpha):
        x = X[:,0]
        n = len(x)
        mu = 1 * np.sin(4 * np.pi * x)
        sigma = np.abs(0.25 * ((1.0-self.a) + 10*self.a*x**2))
        lower = norm.ppf(alpha/2, loc=mu, scale=sigma)
        upper = norm.ppf(1.0-alpha/2, loc=mu, scale=sigma)
        return lower, upper

class Model_Class1:
    def __init__(self, K, p, magnitude=1):
        self.K = K
        self.p = p
        self.magnitude = magnitude
        # Generate model parameters
        self.beta_Z = self.magnitude*np.random.randn(self.p,self.K)

    def sample_X(self, n):
        X = np.random.normal(0, 1, (n,self.p))
        factor = 0.2
        X[0:int(n*factor),0] = 1
        X[int(n*factor):,0] = -8
        return X.astype(np.float32)
    
    def compute_prob(self, X):
        f = np.matmul(X,self.beta_Z)
        prob = np.exp(f)
        prob_y = prob / np.expand_dims(np.sum(prob,1),1)
        return prob_y

    def sample_Y(self, X):
        prob_y = self.compute_prob(X)
        g = np.array([np.random.multinomial(1,prob_y[i]) for i in range(X.shape[0])], dtype = float)
        classes_id = np.arange(self.K)
        y = np.array([np.dot(g[i],classes_id) for i in range(X.shape[0])], dtype = int)
        return y.astype(np.int64)


class Model_Class2:
    def __init__(self, K, p, magnitude=None):
        self.K = K
        self.p = p
        
    def sample_X(self, n):
        X = np.random.normal(0, 1, (n,self.p))
        X[:,0] = np.random.choice([-1,1], size=n, replace=True, p=[1/self.K, (self.K-1)/self.K])
        X[:,1] = np.random.choice([-1,1], size=n, replace=True, p=[1/4, 3/4])
        X[:,2] = np.random.choice([-1,1], size=n, replace=True, p=[1/2, 1/2])
        X[:,3] = np.random.choice(self.K, n, replace=True)
        return X.astype(np.float32)
        
    def compute_prob(self, X):
        prob_y = np.zeros((X.shape[0], self.K))
        right_0 = X[:,0] > 0
        right_1 = X[:,1] > 0
        right_2 = X[:,2] > 0
        leaf_0 = np.where(1-right_0)[0]
        leaf_1 = np.where((right_0) * (1-right_1) * (1-right_2))[0]
        leaf_2 = np.where((right_0) * (1-right_1) * (right_2))[0]
        leaf_3 = np.where((right_0) * (right_1))[0]
        # Zeroth leaf: uniform distribution over all labels
        prob_y[leaf_0] = 1.0/self.K
        # First leaf: uniform distribution over the first half of the labels
        K_half = int(np.round(self.K/2))
        prob_y[leaf_1, 0:K_half] = 2.0/self.K
        prob_y[leaf_1, K_half:self.K] = 0
        # Second leaf: uniform distribution over the second half of the labels
        prob_y[leaf_2, 0:K_half] = 0
        prob_y[leaf_2, K_half:self.K] = 2.0/self.K
        # Third leaf: 90% probability to label determined by 4th variable
        X3 = np.round(X[leaf_3,3]).astype(int)
        for k in range(self.K):
            prob_y[leaf_3[X3==k],:] = (1-0.9)/(self.K-1.0)
            prob_y[leaf_3[X3==k],k] = 0.9
        # Standardize probabilities for each sample        
        prob_y = prob_y / prob_y.sum(axis=1)[:,None]
        return prob_y

    def sample_Y(self, X):
        prob_y = self.compute_prob(X)
        g = np.array([np.random.multinomial(1,prob_y[i]) for i in range(X.shape[0])], dtype = float)
        classes_id = np.arange(self.K)
        y = np.array([np.dot(g[i],classes_id) for i in range(X.shape[0])], dtype = int)
        return y.astype(np.int64)
