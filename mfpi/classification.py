import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ProbabilityAccumulator:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.prob_sort = -np.sort(-prob, axis=1)
        self.Z = np.round(self.prob_sort.cumsum(axis=1),9)        
        
    def predict_sets(self, alpha, randomize=False, epsilon=None):
        if alpha>0:
            L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        else:
            L = (self.Z.shape[1]-1)*np.ones((self.Z.shape[0],)).astype(int)
        if randomize:
            epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        if epsilon is not None:
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.prob_sort[i, L[i]] for i in range(self.n) ])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                L[i] = L[i] - 1
        # Return prediction set
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        return(S)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        prob_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        prob = np.array([ self.prob_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        tau_min = 1-alpha_max
        return tau_min
    
def wsc(X, y, S, delta=0.1, M=1000, verbose=False):

    def wsc_v(X, y, S, delta, v):
        n = len(y)
        cover = np.array([y[i] in S[i] for i in range(n)])
        z = np.dot(X,v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        if bi_best==len(z_sorted):
            bi_best = bi_best-1
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, S, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, S, delta, V[m])
        
    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star

def wsc_unbiased(X, y, S, delta=0.1, M=1000, test_size=0.75, random_state=2020, verbose=False):
    def wsc_vab(X, y, S, v, a, b):
        n = len(y)
        cover = np.array([y[i] in S[i] for i in range(n)])
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(cover[idx])
        return coverage

    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=test_size,
                                                                         random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, y_train, S_train, delta=delta, M=M, verbose=verbose)
    # Estimate coverage
    coverage = wsc_vab(X_test, y_test, S_test, v_star, a_star, b_star)
    return coverage
