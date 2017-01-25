from dolfin import *
from pylab import deg2rad,plot,show,linspace,ones,zeros,copy,array

# VERTICAL BASIS REPLACES A NORMAL FUNCTION, SUCH THAT VERTICAL DERIVATIVES
# CAN BE EVALUATED IN MUCH THE SAME WAY AS HORIZONTAL DERIVATIVES.  IT NEEDS
# TO BE SUPPLIED A LIST OF FUNCTIONS OF SIGMA THAT MULTIPLY EACH COEFFICIENT.
class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

# SIMILAR TO ABOVE, BUT FOR CALCULATION OF FINITE DIFFERENCE QUANTITIES.
class VerticalFDBasis(object):
    def __init__(self,u,deltax,sigmas):
        self.u = u 
        self.deltax = deltax
        self.sigmas = sigmas

    def __call__(self,i):
        return self.u[i]

    def eval(self,s):
        fl = max(sum(s>self.sigmas)-1,0)
        dist = s - self.sigmas[fl]
        return self.u[fl]*(1 - dist/self.deltax) + self.u[fl+1]*dist/self.deltax

    def ds(self,i):
        return (self.u[i+1] - self.u[i-1])/(2*self.deltax)

    def d2s(self,i):
        return (self.u[i+1] - 2*self.u[i] + self.u[i-1])/(self.deltax**2)

    def dx(self,i,x):
        return self.u[i].dx(x)        

# PERFORMS GAUSSIAN QUADRATURE FOR ARBITRARY FUNCTION OF SIGMA, QUAD POINTS, AND WEIGHTS
class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

from numpy.polynomial.legendre import leggauss
def half_quad(order):
    points,weights = leggauss(order)
    points=points[(order-1)/2:]
    weights=weights[(order-1)/2:]
    weights[0] = weights[0]/2
    return points,weights


