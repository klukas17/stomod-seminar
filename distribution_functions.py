import numpy as np

exponential = lambda scale: lambda: np.random.exponential(scale=scale, size=1)[0]
lognormal = lambda mean, sigma: lambda: np.random.lognormal(mean=mean, sigma=sigma, size=1)[0]
weibull = lambda scale, a: lambda: scale * np.random.weibull(a=a, size=1)[0]
wald = lambda mean, scale: lambda: np.random.wald(mean=mean, scale=scale, size=1)[0]
gamma = lambda shape, scale: lambda: np.random.gamma(shape=shape, scale=scale, size=1)[0]
uniform = lambda low, high: lambda: np.random.uniform(low=low, high=high, size=1)[0]
triangular = lambda left, mode, right: lambda: np.random.triangular(left=left, mode=mode, right=right, size=1)[0]
constant = lambda const: lambda: const
