import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv(r"C:\Users\Zenitsu\Downloads\3Fringe_Pixel_radius=4 - Sheet1.csv")

df.drop_duplicates(subset="luma",keep="first",inplace=True)

def func(x,a,b,c,d):
    return(d+a*(np.sin(b*x+c))**2)

a,b=curve_fit(func,df["t"],df["luma"],p0=[80,2,100,5])

t=np.linspace(0,max(df["t"]),1000)
f = func(t,a[0],a[1],a[2],a[3])

plt.plot(df["t"],df["luma"],"r.",label="data points")
plt.plot(t,f,label="curve fit")
plt.xlabel("time")
plt.ylabel("luma")
plt.title("Intensity vs time 2nd fringe(pixel radius = 5) ")
plt.legend()
plt.ylim([min(df["luma"])-10,140])
#plt.show()
m=max(df["luma"])
n=min(df["luma"])
print((m-n)/(m+n))
print(b[0][0]**0.5,b[1][1]**0.5,b[2][2]**0.5,b[3][3]**0.5)