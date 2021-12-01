import numpy, scipy, matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

df = pd.read_csv(r"C:\Users\Zenitsu\Downloads\MI_He-ne - Sheet1.csv")

z = df['t']

r = df["luma"]

xData = z
yData = r


def func (x, amplitude, center, width, offset):
 return amplitude * numpy.sin(numpy.pi * (x - center) / width)**2 + offset


#minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore")
    val = func(xData, *parameterTuple)
    return numpy.sum((yData - val) ** 2.0)


def generate_Initial_Parameters():
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)

    diffY = maxY - minY
    diffX = maxX - minX

    parameterBounds = []
    parameterBounds.append([0.0, diffY]) # search bounds for amplitude
    parameterBounds.append([minX, maxX]) # search bounds for center
    parameterBounds.append([0.0, diffX]) # search bounds for width
    parameterBounds.append([minY, maxY]) # search bounds for offset

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

# by default, differential_evolution completes by calling curve_fit() using parameter bounds
Parameters = generate_Initial_Parameters()

fittedParameters, pcov = curve_fit(func, xData, yData, Parameters)
print('Fitted parameters:', fittedParameters)
print(pcov[0][0]**0.5,pcov[1][1]**0.5,pcov[2][2]**0.5,pcov[3][3]**0.5)

modelPredictions = func(xData, *fittedParameters) 





f = plt.figure(figsize=(8,6), dpi=100)
axes = f.add_subplot(111)

# first the raw data as a scatter plot
axes.plot(xData, yData,  '.-',color="red",label="Data Points")

# create data for the fitted equation plot
xModel = numpy.linspace(min(xData), max(xData))
yModel = func(xModel, *fittedParameters)

# now the model as a line plot
axes.plot(xModel, yModel,color="green",label="Fitted Curve")

axes.set_ylim([min(r),max(r)+5])

axes.set_xlabel('Time (s)') # X axis data label
axes.set_ylabel('Luma Value') # Y axis data label
axes.set_title("Intensity vs Time")
print("time:",max(z))
print("Fringe Contrast:", ((max(r)-min(r))/(max(r)+min(r))))
plt.legend()
plt.show()
plt.close('all') # clean up after using pyplot