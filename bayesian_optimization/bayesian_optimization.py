#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
f_noise_var = 0.005
#%%
def f_original(x:np.ndarray):
    return np.log(x[0])/(np.log(x[1]) + 1.0)

def f_noisy(x:np.ndarray):
    y = f_original(x)
    noise = np.random.normal(loc=0, scale=np.sqrt(f_noise_var), size=y.shape)
    return (y + noise)
#%%
# X = np.zeros((100, 2), dtype=float)
# X[:, 0] = np.linspace(1, 20, 100)
# X[:, 1] = np.linspace(1, 20, 100)

# x0m, x1m = np.meshgrid(X[:, 0], X[:, 1])
# f_orig  = f_original(np.array([x0m, x1m]))
# f_noise = f_noisy(np.array([x0m, x1m]))

# # plot them and see
# plt.figure(figsize=(12,7))
# ax = plt.axes(projection='3d')
# ax.plot_surface(x0m, x1m, f_orig)
# ax.set_xlabel('x0', fontsize=12)
# ax.set_ylabel('x1', fontsize=12)
# ax.set_zlabel(f"f-original, f_max: {np.max(f_orig):0.3f}", fontsize=12)
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(12,7))
# ax = plt.axes(projection='3d')
# ax.plot_surface(x0m, x1m, f_noise)
# ax.set_xlabel('x0', fontsize=12)
# ax.set_ylabel('x1', fontsize=12)
# ax.set_zlabel("f-noisy", fontsize=12)
# plt.grid(True)
# plt.show()

#%% Define Gaussian Process
class BayesOptim:
    x_limits_ = None
    f_        = None
    x_hist_ = np.array([])
    f_hist_ = np.array([])
    kernel_params_ = {}
    n_iter_ = 0

    k11_     = np.array([])
    k11_inv_ = np.array([])

    x_best_ = None
    f_best_ = -np.inf
    def __init__(self, f, x_limits, kernel_params=None):
        self.f_ = f
        self.x_limits_ = x_limits
        if (kernel_params == None):
            self.kernel_params_['alpha'] = 0.8
            self.kernel_params_['lambda'] = 0.9
        else:
            self.kernel_params_ = kernel_params
    
    # Squared-Exponential Kernel
    def kernel(self, x1, x2):
        norm = np.linalg.norm(x1 - x2)**2
        cov  = self.kernel_params_['alpha']*np.exp(-0.5*norm/self.kernel_params_['lambda'])
        return cov

    def calc_k11_(self):
      if (self.k11_.size == 0):
        self.k11_ = np.zeros((self.n_iter_, self.n_iter_), dtype=float)

        for i in range(0, self.k11_.shape[0]):
          # copy already calculated cross covariance terms from previous rows
          for k in range(0, i):
              self.k11_[i, k] = self.k11_[k, i]

          # calculate remaining terms    
          for j in range(i, self.k11_.shape[1]):
              self.k11_[i, j] = self.kernel(self.x_hist_[i], self.x_hist_[j])
      else:
        # new last column
        new_elems = np.zeros((self.n_iter_, 1), dtype=float)
        for i in range(0, self.n_iter_):
          new_elems[i, 0] = self.kernel(self.x_hist_[i], self.x_hist_[-1])

        self.k11_ = np.hstack((self.k11_, new_elems[0:-1]))
        self.k11_ = np.vstack((self.k11_, new_elems.T))
      
      self.k11_inv_ = np.linalg.inv(self.k11_)

    def calc_proposal_cov_(self, x_proposal):
      k12 = np.zeros( (self.n_iter_, x_proposal.shape[0]), dtype=float)
      k22 = np.zeros( (x_proposal.shape[0], x_proposal.shape[0]), dtype=float)
      # calculate K12
      for i in range(0, k12.shape[0]):
        for j in range(0, k12.shape[1]):
          k12[i, j] = self.kernel(self.x_hist_[i], x_proposal[j])
    
      # calculate k22
      for i in range(0, k22.shape[0]):
        # copy already calculated cross covariance terms from previous rows
        for k in range(0, i):
          k22[i, k] = k22[k, i]   
        for j in range(i, k22.shape[1]):
          k22[i, j] = self.kernel(x_proposal[i], x_proposal[j])

      return k12, k22

    def gpr_stats(self, k11, k11_inv, k12, k22, f_hist):
      k21 = np.transpose(k12)
      mu  = k21 @ k11_inv @ f_hist
      var = k22 - (k21 @ k11_inv @ k12)
      sigma = np.sqrt(np.diag(var))
      
      return mu, sigma
    
    def ucb(self, mu, sigma, l=100.0):
      val = mu + l*sigma
      return val

    def acquisition_optimize(self, X0):
      grad = np.zeros((X0.shape[0]), dtype=float)

      for i in range(0, X0.shape[0]):
        x_forward = X0
        x_backward = X0
        x_forward[i] += 1e-5
        x_backward[i] -= 1e-5

        # evalulate each and get the best sample to process
        k12, k22 = self.calc_proposal_cov_(x_forward)
        mu, sigma = self.gpr_stats(self.k11_, self.k11_inv_, k12, k22, self.f_hist_)

        mu      = mu.reshape(n_samples, 1)
        sigma   = sigma.reshape(n_samples, 1)
        ucb_val_forward = self.ucb(mu, sigma)

        k12, k22 = self.calc_proposal_cov_(x_backward)
        mu, sigma = self.gpr_stats(self.k11_, self.k11_inv_, k12, k22, self.f_hist_)

        mu      = mu.reshape(n_samples, 1)
        sigma   = sigma.reshape(n_samples, 1)
        ucb_val_backward = self.ucb(mu, sigma)

        grad[i] = (ucb_val_forward - ucb_val_backward)/(2.0*1e-5)

      
    def acquisition(self):
      n_samples = 100
      samples_2_eval = []
      for i in range(0, self.x_limits_.shape[0]):
          x = np.random.random_sample(n_samples) * (self.x_limits_[i, 1] - self.x_limits_[i, 0]) + self.x_limits_[i, 0]
          samples_2_eval.append(x)
      samples_2_eval = np.array(samples_2_eval).T

      # evalulate each and get the best sample to process
      k12, k22 = self.calc_proposal_cov_(samples_2_eval)
      mu, sigma = self.gpr_stats(self.k11_, self.k11_inv_, k12, k22, self.f_hist_)

      mu      = mu.reshape(n_samples, 1)
      sigma   = sigma.reshape(n_samples, 1)
      ucb_val = self.ucb(mu, sigma)
      
      i_best = np.argmax(ucb_val)
      return samples_2_eval[i_best]
    
    def fit(self, x, f):
      self.x_hist_ = np.vstack((self.x_hist_, x))
      self.f_hist_ = np.vstack((self.f_hist_, f))

      # calculate covariances
      self.calc_k11_()

    def bootstrap(self):
      # first evaluate f by drawing x randomly to bootstrap the optimization
      for i in range(0, 3):
          samples_2_eval = []
          for i in range(0, self.x_limits_.shape[0]):
              x = np.random.random_sample(1) * (self.x_limits_[i, 1] - self.x_limits_[i, 0]) + self.x_limits_[i, 0]
              samples_2_eval.append(x)
          samples_2_eval = (np.array(samples_2_eval).T)[0]
          
          f = self.f_(samples_2_eval)
          
          if (self.f_hist_.size == 0):
            self.f_hist_ = f
            self.x_hist_ = samples_2_eval
          else:
            self.f_hist_ = np.vstack( (self.f_hist_, f) )
            self.x_hist_ = np.vstack( (self.x_hist_, samples_2_eval) )
          self.n_iter_ += 1
      self.calc_k11_()

    def optimize(self):
      self.bootstrap()
      while (self.n_iter_ <= 100):
        x_proposal = self.acquisition()
        f = self.f_(x_proposal)

        if (f > self.f_best_):
          self.f_best_ = f
          self.x_best_ = x_proposal

        self.n_iter_ += 1
        self.fit(x_proposal, f)
        print(f"iteration: {self.n_iter_}, f: {f}, f_best: {self.f_best_}")

      return self.x_best_, self.f_best_

#%%
X_limits = np.array([ [1.0, 20.0], [1.0, 20.0] ])
bo = BayesOptim(f_original, X_limits)
x_best, f_best = bo.optimize()

print(f"BayesOptim: {f_best}")
