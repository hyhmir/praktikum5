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

print(f"------------------------------\nfrekvenca: {freq} GHz")
print('--------------------------------')


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

s_b = umath.sqrt(maxi/mini) # type: ignore
print(f'Ubranost = {s_b}')

plt.plot(premik(df['CH1'], False), df['CH2'], '.',  ms=3, label='Bolometer')
# plt.show()


##### antena #####


dfi = pd.read_csv('07_uVal/data/NewFile4.csv', skiprows=[1], usecols=["CH1", "CH2"])

df = dfi.rolling(10,1,True).mean()

maxi = ufloat(np.max(df['CH2']), np.abs(np.max(df['CH2']) - np.max(dfi['CH2'])))
mini = ufloat(np.min(df['CH2']), np.abs(np.min(df['CH2']) - np.min(dfi['CH2'])))

s_a = umath.sqrt(maxi/mini) # type: ignore
print(f'Ubranost = {s_a}')

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
# plt.show()
plt.clf()


##### racunanje stvari #####


lamb_ = ufloat(4.962, 0.08)
a = ufloat(2.6, 0.1)
c = 29979245800
fr = c*(umath.sqrt(lamb_**2 + 4*a**2))/(2*a*lamb_) # type: ignore

print(f' ------------------------------\nfrekvenca izracunana iz valovne dolzine: {fr}\n ------------------------------\n')

rR = (s_b-1)**2/(s_b+1)**2

x_b = ufloat(1.178, 0.08)
x_a = ufloat(2.073, 0.12)

bxmb = 2*np.pi*x_b/lamb_ # type: ignore
bxma = 2*np.pi*x_a/lamb_ # type: ignore

njRb = (s_b**2-1)*umath.tan(bxmb)/(1+s_b**2*umath.tan(bxmb)**2) # type: ignore
njRa = (s_a**2-1)*umath.tan(bxma)/(1+s_a**2*umath.tan(bxma)**2) # type: ignore

xjRb = (1 - njRb*umath.tan(bxmb))*s_b # type: ignore
xjRa = (1 - njRa*umath.tan(bxma))*s_a # type: ignore

ib = umath.sqrt(njRb**2 + xjRb**2) # type: ignore
ia = umath.sqrt(njRa**2 + xjRa**2) # type: ignore

print(f'reaktanca in rezistenca sta: {njRb} in {xjRb} za bolometer {ib}')
print(f'reaktanca in rezistenca sta: {njRa} in {xjRa} za anteno {ia}')


##### moc rodov #####

df = pd.read_csv('07_uVal/data/NewFile9.csv')

df['P actual'] = df['P']
df['P actual err'] = df['P']

for i in range(len(df['P'])):
    pm = ufloat(df['P'][i], df['P_err'][i])
    p = pm / (1 - umath.sqrt(rR)) # type: ignore
    df.loc[i, 'P actual'] = p.n
    df.loc[i, 'P actual err'] = p.s

# plt.errorbar(df['U'], df['P'], xerr=df['U_err'], yerr=df['P_err'], fmt='.',  ms=3, label='meritve')
plt.errorbar(-df['U'], df['P actual'], xerr=df['U_err'], yerr=df['P actual err'], fmt='.',  ms=3, label='meritve')
plt.grid()
plt.title('Rodovi klistrona')
plt.ylabel('P [mW]')
plt.xlabel('U [V]')
plt.savefig('07_uVal/porocilo/moc.pdf', dpi=512)
plt.clf()
# plt.show()

##### prikaz razmazanosti ######

dfi = pd.read_csv('07_uVal/data/NewFile6.csv', skiprows=[1], usecols=["CH1", "CH2"])
plt.plot(premik(dfi['CH1'], False), dfi['CH2']/10, '.')
dfi = pd.read_csv('07_uVal/data/NewFile4.csv', skiprows=[1], usecols=["CH1", "CH2"])
plt.plot(premik(dfi['CH1'], False), dfi['CH2'], '.')
dfi = pd.read_csv('07_uVal/data/NewFile2.csv', skiprows=[1], usecols=["CH1", "CH2"])
plt.plot(premik(dfi['CH1'], False), dfi['CH2'], '.')
plt.grid()
plt.title('Ponazoritev razmazanosti')
plt.savefig('07_uVal/porocilo/razmazanost.pdf', dpi=512)
plt.clf()