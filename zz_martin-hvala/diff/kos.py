import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData

# Nastavitve za Matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size" : 16,
    "font.sans-serif": ["Helvetica"]})


## Meritve

t0 = 37.9e-6    # Čas na razdelek

t2 = np.arange(0, 51, 5) * t0
A2 = np.array([1.60, 1.20, 0.96, 0.60, 0.37, 0.27, 0.19, 0.14, 0.1, 0.09, 0.06])

del_t = 5e-5
del_A = 0.1

del_t = t0 * 0.5 
del_A = 0.1

for t, A in zip(t2, A2):
    a = np.random.uniform(0, 0.02)
    b = np.random.uniform(0, 0.02)
    a = b = 0
    print(
        f'{1e3 *t + a:.2f} {A + b:.2f}'
    )
