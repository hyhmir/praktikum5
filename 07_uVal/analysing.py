import numpy as np
import pandas as pd
from uncertainties import ufloat
import matplotlib.pyplot as plt
import scienceplots
from uncertainties import umath

plt.style.use('science')


##### frekvenca #####

res_poz = ufloat(413.0, 0.5)
freq = -2/400 * res_poz + 10.5 # type: ignore

print(f"frekvenca: {freq} GHz")
print('\n \n --------------------------------')


##### kalibracija #####

df = pd.read_csv('07_uVal/data/NewFile7.csv', skiprows=[1], usecols=["CH1", "CH2"])
min_V = ufloat(df['CH1'].mean(), df['CH1'].std())
# print(min_V)


df = pd.read_csv('07_uVal/data/NewFile8.csv', skiprows=[1], usecols=["CH1", "CH2"])
max_V = ufloat(df['CH1'].mean(), df['CH1'].std())
# print(max_V)

def premik(V, err=True):
    if err:
        m = ufloat(5.7, 0.2)/(max_V - min_V) # type: ignore
        b = ufloat(2.2, 0.1) - m * min_V # type: ignore
        return m * V + b # type: ignore
    m = ufloat(5.7, 0.2).n/(max_V.n - min_V.n) # type: ignore
    b = 2.2 - m * min_V.n # type: ignore
    return m * V + b # type: ignore


##### bolometer #####

dfi = pd.read_csv('07_uVal/data/NewFile2.csv', skiprows=[1], usecols=["CH1", "CH2"])

df = dfi.rolling(10,1,True).mean()

maxi = ufloat(np.max(df['CH2']), np.abs(np.max(df['CH2']) - np.max(dfi['CH2'])))
mini = ufloat(np.min(df['CH2']), np.abs(np.min(df['CH2']) - np.min(dfi['CH2'])))

s = umath.sqrt(maxi/mini) # type: ignore
print(f'Ubranost = {s}')

plt.plot(premik(df['CH1'], False), df['CH2'], '.',  ms=3, label='Bolometer')
# plt.show()


##### antena #####


dfi = pd.read_csv('07_uVal/data/NewFile4.csv', skiprows=[1], usecols=["CH1", "CH2"])

df = dfi.rolling(10,1,True).mean()

maxi = ufloat(np.max(df['CH2']), np.abs(np.max(df['CH2']) - np.max(dfi['CH2'])))
mini = ufloat(np.min(df['CH2']), np.abs(np.min(df['CH2']) - np.min(dfi['CH2'])))

s = umath.sqrt(maxi/mini) # type: ignore
print(f'Ubranost = {s}')

plt.plot(premik(df['CH1'], False), df['CH2'], '.',  ms=3, label='Antena')
# plt.show()


##### kratkosticna stena #####


dfi = pd.read_csv('07_uVal/data/NewFile6.csv', skiprows=[1], usecols=["CH1", "CH2"])

df = dfi.rolling(10,1,True).mean()

maxi = ufloat(np.max(df['CH2']), np.abs(np.max(df['CH2']) - np.max(dfi['CH2'])))
mini = ufloat(np.min(df['CH2']), np.abs(np.min(df['CH2']) - np.min(dfi['CH2'])))

s = umath.sqrt(maxi/mini) # type: ignore
print(f'Ubranost = {s}')

plt.plot(premik(df['CH1'], False), df['CH2']/10, '.',  ms=3, label='Kratkostična stena $/10$')
plt.legend()
plt.grid()
plt.title('Krivulja ubranosti')
plt.xlabel('x [cm]')
plt.ylabel('U [V]')
plt.savefig('07_uVal/porocilo/ubranost.pdf', dpi=512)
plt.show()
plt.clf()


##### moc rodov #####

df = pd.read_csv('07_uVal/data/NewFile9.csv')

plt.errorbar(df['U'], df['P'], xerr=df['U_err'], yerr=df['P_err'], fmt='.',  ms=3, label='meritve')
plt.clf()
# plt.show()