
import pandas as pd
from uncertainties import ufloat
import matplotlib.pyplot as plt


##### frekvenca #####

res_poz = ufloat(413.0, 0.5)
freq = -2/400 * res_poz + 10.5 # type: ignore

print(f"frekvenca: {freq} GHz")
print('\n \n --------------------------------')


##### moc rodov #####

df = pd.read_csv('07_uVal/data/NewFile9.csv')

plt.errorbar(df['U'], df['P'], xerr=df['U_err'], yerr=df['P_err'], fmt='.', label='meritve')
plt.clf()
# plt.show()

##### kalibracija #####

df = pd.read_csv('07_uVal/data/NewFile7.csv', skiprows=[1], usecols=["CH1", "CH2"])
min_V = ufloat(df['CH1'].mean(), df['CH1'].std())
print(min_V)


df = pd.read_csv('07_uVal/data/NewFile8.csv', skiprows=[1], usecols=["CH1", "CH2"])
max_V = ufloat(df['CH1'].mean(), df['CH1'].std())
print(max_V)

def premik(V, err=True):
    if err:
        m = ufloat(5.7, 0.2)/(max_V - min_V) # type: ignore
        b = ufloat(2.2, 0.1) - m * min_V # type: ignore
        return m * V + b # type: ignore
    m = ufloat(5.7, 0.2).n/(max_V.n - min_V.n) # type: ignore
    b = 2.2 - m * min_V.n # type: ignore
    return m * V + b # type: ignore


##### bolometer #####

df = pd.read_csv('07_uVal/data/NewFile2.csv', skiprows=[1], usecols=["CH1", "CH2"])
plt.plot(premik(df['CH1'], False), df['CH2'], '.', label='meritve')
# plt.show()


##### antena #####

df = pd.read_csv('07_uVal/data/NewFile4.csv', skiprows=[1], usecols=["CH1", "CH2"])
plt.plot(premik(df['CH1'], False), df['CH2'], '.', label='meritve')
# plt.show()


##### kratkosticna stena #####

df = pd.read_csv('07_uVal/data/NewFile6.csv', skiprows=[1], usecols=["CH1", "CH2"])
plt.plot(premik(df['CH1'], False), df['CH2'], '.', label='meritve')
plt.show()
