import numpy
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import math

# Create 1D Gaussian toy data.
numpy.random.seed(1) #set random seed
# Draw 10 values from unit Gaussian.
Data = numpy.random.normal(0.0, 1.0, 10)
# Range of Parameter a.
a_min = -2.5
a_max =  2.5
# Range of parameter b.
b_min = -1.0
b_max =  1.0
# Number of steps of grid.
Steps = 51
# Allocate grid as matrix.
Grid = numpy.zeros([Steps,Steps])
# Try all parameter combinations.
for s1 in range(Steps):
    for s2 in range(Steps):
        # Current parameter combination.
        a = a_min + (a_max -a_min)*float(s1)/float(Steps-1)
        b = b_min + (b_max -b_min)*float(s2)/float(Steps-1)

        # Evaluate chi-squared.
        chi2 = 0.0
        for n in range(len(Data)):
            # Use index n as pseudo-position
            residual = (Data[n] - a - n*b)
            chi2     = chi2 + residual*residual
        Grid[Steps-1-s2,s1] = chi2

plt.figure(1, figsize=(8,3))
mini  = numpy.min(Grid) # minimal value of chi2
image = plt.imshow(Grid, vmin=mini, vmax=mini+20.0, extent=[a_min,a_max,b_min,b_max])
plt.colorbar(image)
plt.xlabel(r'$a$', fontsize=24)
plt.ylabel(r'$b$', fontsize=24)
plt.savefig('example-chi2-mainfold.png')
plt.show()

#---

# Generate artificial data = straight line with a=0 and b=1
# plus some noise.
xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])
# Initial guess.
x0    = numpy.array([0.0, 0.0, 0.0])

sigma = numpy.array([1.0,1.0,1.0,1.0,1.0,1.0])

def func(x, a, b, c):
    return a + b*x + c*x*x

print(optimization.curve_fit(func, xdata, ydata, x0, sigma))

#---

# The function whose square is to be minimised.
# params ... list of parameters tuned to minimise function.
# Further arguments:
# xdata ... design matrix for linear model.
# ydata ... observed data.
def func1(params, xdata, ydata):
    return (ydata - numpy.dot(xdata, params))

# Provide data as design matrix: straight line with a=0 and b=1 plus some noise.
x0 = numpy.array([0.0,0.0])
xdata = numpy.transpose(numpy.array([[1.0,1.0,1.0,1.0,1.0,1.0],[0.0,1.0,2.0,3.0,4.0,5.0]]))

print(optimization.leastsq(func1, x0, args=(xdata, ydata)))

#---

# Chose a model that will create bimodality.
def func2(x, a, b):
    return a + b*b*x # Term b*b will create bimodality.

# Create toy data for curve_fit.
xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])
sigma = numpy.array([1.0,1.0,1.0,1.0,1.0,1.0])

# Compute chi-square manifold.
Steps = 101 # grid size
Chi2Mainfold = numpy.zeros([Steps,Steps]) # allocate grid
amin = -7.0 # minimal value of a covered by grid
amax = +5.0 # maximal value of a covered by grid
bmin = -4.0 # minimal value of b covered by grid
bmax = +4.0 # maximal value of b covered by grid
for s1 in range(Steps):
    for s2 in range(Steps):
        # Current values of (a,b) at grid position (s1,s2).
        a = amin + (amax - amin)*float(s1)/(Steps-1)
        b = bmin + (bmax - bmin)*float(s2)/(Steps-1)
        # Evaluate chi-squared.
        chi2 = 0.0
        for n in range(len(xdata)):
            residual = (ydata[n] - func2(xdata[n], a, b))/sigma[n]
            chi2 = chi2 + residual*residual
        Chi2Mainfold[Steps-1-s2,s1] = chi2 # write result to grid.

# Plot grid.
plt.figure(1, figsize=(8,4.5))
plt.subplots_adjust(left=0.09, bottom=0.09, top=0.97, right=0.99)
# Plot chi-square mainfold.
image = plt.imshow(Chi2Mainfold, vmax=50.0, extent=[amin, amax, bmin, bmax])
# Plot where curve-fit is going to for a couple of initial guesses.
for a_initial in -6.0, -4.0, -2.0, 0.0, 2.0, 4.0:
    #Initial guess
    x0   = numpy.array([a_initial, -3.5])
    xFit = optimization.curve_fit(func2, xdata, ydata, x0, sigma)[0]
    plt.plot([x0[0], xFit[0]], [x0[1], xFit[1]], 'o-', ms=4,markeredgewidth=0, lw=2, color='orange')
plt.colorbar(image) # make colorbar
plt.xlim(amin, amax)
plt.ylim(bmin, bmax)
plt.xlabel(r'$a$', fontsize=24)
plt.ylabel(r'$b$', fontsize=24)
plt.savefig('demo-robustness-curve_fit.png')
plt.show()
