from pylab import *
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage as nd

import numpy as np
import gdal

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

###  DATA  ###
data    = gdal.Open('input_data_bed_v2/DEM_2010/ifsar_2010.tif')
S_array = data.ReadAsArray()[::-1,:]
ncols = data.RasterXSize
nrows = data.RasterYSize

transf = data.GetGeoTransform()

x = arange(transf[0],transf[0]+transf[1]*data.RasterXSize,transf[1])
y = arange(transf[3],transf[3]+transf[5]*data.RasterYSize,transf[5])[::-1]

S_spline = RectBivariateSpline(x,y,S_array.T,kx=1,ky=1,s=0)

data    = gdal.Open('input_data_bed_v2/BED_MC/bed.tif')
B_array = data.ReadAsArray()[::-1,:]
ncols = data.RasterXSize
nrows = data.RasterYSize

transf = data.GetGeoTransform()

x = arange(transf[0],transf[0]+transf[1]*data.RasterXSize,transf[1])
y = arange(transf[3],transf[3]+transf[5]*data.RasterYSize,transf[5])[::-1]

B_spline = RectBivariateSpline(x,y,B_array.T,kx=1,ky=1,s=0)

data    = gdal.Open('input_data_bed_v2/SMB_2010_2013/mb_field_25.tif')
adot_array = data.ReadAsArray()[::-1,:]
adot_array = fill(adot_array,adot_array==adot_array.min())
ncols = data.RasterXSize
nrows = data.RasterYSize

transf = data.GetGeoTransform()

x = arange(transf[0],transf[0]+transf[1]*data.RasterXSize,transf[1])
y = arange(transf[3],transf[3]+transf[5]*data.RasterYSize,transf[5])[::-1]

adot_spline = RectBivariateSpline(x,y,adot_array.T,kx=1,ky=1,s=0)

data    = gdal.Open('input_data_bed_v2/DH_2010_2013/dhdt_weq_lower.tif')
dhdt_array = data.ReadAsArray()[::-1,:]
dhdt_array[dhdt_array<-1000] = 0
dhdt_array = fill(dhdt_array,dhdt_array==dhdt_array.min())
ncols = data.RasterXSize
nrows = data.RasterYSize

transf = data.GetGeoTransform()

x = arange(transf[0],transf[0]+transf[1]*data.RasterXSize,transf[1])
y = arange(transf[3],transf[3]+transf[5]*data.RasterYSize,transf[5])[::-1]

dhdt_spline = RectBivariateSpline(x,y,dhdt_array.T,kx=1,ky=1,s=0)

from dolfin import *
from ice_model_functions import *

##########################################################
#################  SET PETSC OPTIONS  ####################
##########################################################

PETScOptions.set("ksp_type","preonly")
PETScOptions.set("pc_type","lu")
PETScOptions.set("pc_factor_mat_solver_package","mumps")
PETScOptions.set("mat_mumps_icntl_14","1000")
PETScOptions.set("ksp_final_residual","0")

##########################################################
#################  SET FENICS OPTIONS  ###################
##########################################################

parameters['form_compiler']['quadrature_degree'] = 2
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['representation'] = 'quadrature'
#parameters['form_compiler']['precision'] = 30
parameters['allow_extrapolation'] = True

ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


##########################################################
####################  CONSTANTS  #########################
##########################################################

# TIME
minute = 60.0
hour = 60*minute
day = 24*hour
year = 365*day

# CONSTANTS
rho = 917.
g = 9.81

# RHEOLOGICAL CONSTANTS
rho_i = 910.
n = 3.0

Bc = 3.61e-13*year
Bw = 1.73e3*year
Qc = 6e4
Qw = 13.9e4
Rc = 8.314
gamma = 8.7e-4

eps_reg = Constant(1e-10)

# THERMAL CONTANTS
k = 2.1*year
Cp = 2009.
kappa = k/(rho_i*Cp)
q_geo = 0.042*year

# ADJOINT REG
theta = Constant(1e-10)

# MASS
thklim = 10.
dt = Constant(0.001)

###################################################
########### GEOMETRY AND INPUT DATA  ##############
###################################################


##### BOUNDARY DATA #####
class Beta2(Expression):
  def eval(self,values,x):
    values[0] = 11000.

class S_exp(Expression):
  def eval(self,values,x):
    values[0] = S_spline(x[0],x[1])

class B_exp(Expression):
  def eval(self,values,x):
    values[0] = B_spline(x[0],x[1])

class Adot_exp(Expression):
  def eval(self,values,x):
    values[0] = 1000.0/910.0*adot_spline(x[0],x[1])/3. # Christian provides these fields as mwe/3a, hence correction.

class Dhdt_exp(Expression):
  def eval(self,values,x):
    values[0] = 1000.0/910.0*dhdt_spline(x[0],x[1])/3.

mesh = Mesh('outline.xml')

# FUNCTION SPACES
Q = FunctionSpace(mesh,"CG",1) # SCALAR
Q2 = MixedFunctionSpace([Q,]*2)
V = MixedFunctionSpace([Q]*5) # VELOCITY + MASS

beta2 = interpolate(Beta2(),Q)

#### !!!!!! ####  Note the distinction between effective and normal mass balance !
adot = interpolate(Adot_exp(),Q) - interpolate(Dhdt_exp(),Q)  # Effective mass balance (near steady initially)
#adot = interpolate(Adot_exp(),Q)                             # True mass balance (way imbalanced)
B = interpolate(B_exp(),Q)

S_obs = interpolate(S_exp(),Q) 
S0 = interpolate(S_exp(),Q)

H0 = Function(Q)
H0.vector()[:] = S_obs.vector()[:] - B.vector()[:]  # Set initial thickness

# FUNCTIONS 
U = Function(V)
Lamda = Function(V)
Phi = TestFunction(V)
dU = TrialFunction(V)

gamma = TestFunction(Q)

ubar,vbar,udef,vdef,H = split(U)
phibar,psibar,phidef,psidef,xsi = split(Lamda)

S = B+H

# METRICS FOR COORDINATE TRANSFORM
def dsdx(s):
    return 1./H*(S.dx(0) - s*H.dx(0))

def dsdy(s):
    return 1./H*(S.dx(1) - s*H.dx(1))

def dsdz(s):
    return -1./H

p = 4

# TEST FUNCTION COEFFICIENTS
coef = [lambda s:1.0, lambda s:1./p*((p+1)*s**p - 1)]
dcoef = [lambda s:0, lambda s:(p+1)*s**(p-1)]

u_ = [ubar,udef]
v_ = [vbar,vdef]
phi_ = [phibar,phidef]
psi_ = [psibar,psidef]

u = VerticalBasis(u_,coef,dcoef)
v = VerticalBasis(v_,coef,dcoef)
phi = VerticalBasis(phi_,coef,dcoef)
psi = VerticalBasis(psi_,coef,dcoef)

# TERMWISE STRESSES AND NONLINEARITIES
def A_v():
    return Constant(1e-16)

# 2nd INVARIANT STRAIN RATE
def epsilon_dot(s):
    return ((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                +(v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
                +(u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
                +0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
                + ((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
                + eps_reg)

# VISCOSITY
def eta_v(s):
    return A_v()**(-1./n)/2.*epsilon_dot(s)**((1.-n)/(2*n))

# MEMBRANE STRESSES
E = Constant(1.0)
def membrane_xx(s):
    return (phi.dx(s,0) + phi.ds(s)*dsdx(s))*H*(E*eta_v(s))*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))

def membrane_xy(s):
    return (phi.dx(s,1) + phi.ds(s)*dsdy(s))*H*(E*eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

def membrane_yx(s):
    return (psi.dx(s,0) + psi.ds(s)*dsdx(s))*H*(E*eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

def membrane_yy(s):
    return (psi.dx(s,1) + psi.ds(s)*dsdy(s))*H*(E*eta_v(s))*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))

# SHEAR STRESSES
def shear_xz(s):
    return dsdz(s)**2*phi.ds(s)*H*eta_v(s)*u.ds(s)

def shear_yz(s):
    return dsdz(s)**2*psi.ds(s)*H*eta_v(s)*v.ds(s)

# DRIVING STRESSES
def tau_dx():
    return rho*g*H*S.dx(0)*Lamda[0]

def tau_dy():
    return rho*g*H*S.dx(1)*Lamda[1]

def boundary_membrane_xx(s):
    return phi(s)*H*(E*eta_v(s))*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))

def boundary_membrane_xy(s):
    return phi(s)*H*(E*eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

def boundary_membrane_yx(s):
    return psi(s)*H*(E*eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

def boundary_membrane_yy(s):
    return psi(s)*H*(E*eta_v(s))*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))



N = FacetNormal(mesh)
# GET QUADRATURE POINTS (THIS SHOULD BE ODD: WILL GENERATE THE GAUSS-LEGENDRE RULE 
# POINTS AND WEIGHTS OF O(n), BUT ONLY THE POINTS IN [0,1] ARE KEPT< DUE TO SYMMETRY.
points,weights = half_quad(11)

# INSTANTIATE VERTICAL INTEGRATOR
vi = VerticalIntegrator(points,weights)

# FIRST ORDER EQUATIONS
I_x = - vi.intz(membrane_xx) - vi.intz(membrane_xy) - vi.intz(shear_xz) - phi(1)*beta2*u(1) - tau_dx()
I_y = - vi.intz(membrane_yx) - vi.intz(membrane_yy) - vi.intz(shear_yz) - psi(1)*beta2*v(1) - tau_dy() 


I = (I_x + I_y)*dx

### MASS BALANCE ###
# SUPG PARAMETERS
h = CellSize(mesh)
tau = h/(2.0*sqrt(U[0]**2 + U[1]**2 + 25.0))

Hmid = 0.5*H + 0.5*H0
xsihat = tau*(U[0]*xsi.dx(0) + U[1]*xsi.dx(1))

# STABILIZED CONTINUITY EQUATION
I += ((H - H0)/dt*xsi - (xsi.dx(0)*U[0]*Hmid + xsi.dx(1)*U[1]*Hmid) + xsihat*(U[0]*Hmid.dx(0) + U[1]*Hmid.dx(1) + Hmid*(U[0].dx(0) + U[1].dx(1))) - (adot)*(xsi + xsihat))*dx# + xsi*(U[0]*Hmid*N[0] + U[1]*Hmid*N[1])*ds(1)

I_misfit = theta*dot(grad(beta2),grad(beta2))*dx
I += I_misfit

# JACOBIAN FOR COUPLED MASS + MOMENTUM SOLVE
R = derivative(I,Lamda,Phi)
J = derivative(R,U,dU)

# Adjoint forms, if so desired
R_adj = derivative(I,U,Phi)
J_adj = derivative(R_adj,Lamda,dU)

G = derivative(I,beta2,gamma)

#####################################################################
#########################  I/O Functions  ###########################
#####################################################################

# For moving data between vector functions and scalar functions 
assigner_inv = FunctionAssigner([Q,Q,Q,Q,Q],V)
assigner     = FunctionAssigner(V,[Q,Q,Q,Q,Q])
assigner_vec = FunctionAssigner(Q2,[Q,Q])

#####################################################################
######################  Variational Solvers  ########################
#####################################################################

# Positivity constraints and zero-flux boundary conditions don't play well together, so I enforce the former through a non-slip Dirichlet boundary condition on velocity.  This is a little weird in the context of glaciers, but it's the only condition that will uphold mass conservation (there is still a fictitious momentum flux across the boundary, aka a non-real stress, but that's more acceptable to me).
bcs = [DirichletBC(V.sub(i),0,lambda x,on:on) for i in range(4)]
bc_2 = DirichletBC(V.sub(4),thklim,lambda x,o:(o and x[0]>393092) or (o and (x[1]>1.5273e6 and x[0]<372129 and x[0]>368953)))

mass_problem = NonlinearVariationalProblem(R,U,J=J,bcs=bcs+[bc_2],form_compiler_parameters=ffc_options)

mass_solver = NonlinearVariationalSolver(mass_problem)
mass_solver.parameters['nonlinear_solver'] = 'snes'

mass_solver.parameters['snes_solver']['method'] = 'vinewtonrsls'
mass_solver.parameters['snes_solver']['relative_tolerance'] = 1e-6
mass_solver.parameters['snes_solver']['absolute_tolerance'] = 1e-6
mass_solver.parameters['snes_solver']['maximum_iterations'] = 10
mass_solver.parameters['snes_solver']['error_on_nonconvergence'] = False
mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'

bc_adj_1 = DirichletBC(V,[0.0,0.0,0.0,0.0,0.0],lambda x,on:on)
bc_adj_2 = DirichletBC(V.sub(4),0.0,lambda x,on:on)
adj_problem = NonlinearVariationalProblem(R_adj,Lamda,J=J_adj,bcs=[bc_adj_1,bc_adj_2],form_compiler_parameters=ffc_options)
adj_solver = NonlinearVariationalSolver(adj_problem)

adj_solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
adj_solver.parameters['newton_solver']['absolute_tolerance'] = 1e-3
adj_solver.parameters['newton_solver']['maximum_iterations'] = 3
adj_solver.parameters['newton_solver']['error_on_nonconvergence'] = False
adj_solver.parameters['newton_solver']['linear_solver'] = 'mumps'


#####################################################################
##################  INITIAL CONDITIONS AND BOUNDS  ##################
#####################################################################

l_thick_bound = project(Constant(thklim),Q)
u_thick_bound = project(Constant(1e4),Q)

l_v_bound = project(-10000.0,Q)
u_v_bound = project(10000.0,Q)

l_bound = Function(V)
u_bound = Function(V)

un = Function(Q)
u2n = Function(Q)
vn = Function(Q)
v2n = Function(Q)

lx = Function(Q)
l2x = Function(Q)
mx = Function(Q)
m2x = Function(Q)
p0 = Function(Q)

assigner.assign(U,[un,vn,u2n,v2n,H0])
assigner.assign(l_bound,[l_v_bound]*4+[l_thick_bound])
assigner.assign(u_bound,[u_v_bound]*4+[u_thick_bound])

results_dir = './results/'
Hfile_ptc = File(results_dir + 'H.pvd')
Ufile_ptc = File(results_dir + 'Us.pvd')
bfile_ptc = File(results_dir + 'beta2.pvd')

opt_dir = './results_opt/'
Ufile_opt = File(opt_dir + 'Us.pvd')
bfile_opt = File(opt_dir + 'beta2.pvd')

Us = project(as_vector([u(0),v(0)]))
assigner_inv.assign([lx,l2x,mx,m2x,p0],Lamda)

# Uncomment if you want to start from the end of the last run
#File(results_dir + 'U.xml') >> U 
#H0_temp = project(H)
#H0.vector()[:] = H0_temp.vector()[:]

t = 2016.75

#Start slow for convergence.  Due to an oddity in topography, this model will not converge for the first 5 or so iterations as it fills a hole, then will work fine after.
dt_schedule = [0.00001]*5 + [0.01]*10 + [0.1]*5 + [0.5]*100 + [1.0]*100

#Time stepping
solve(R==0, U, bcs=bcs+[bc_2])
assigner_inv.assign([un,vn,u2n,v2n,H0],U)

#H0_temp = project(H)
#H0.vector()[:] = H0_temp.vector()[:]
    
Us_temp = project(as_vector([u(0),v(0)]))
Us.vector()[:] = Us_temp.vector()[:]
    
S_temp = project(S)
S0.vector()[:] = S_temp.vector()[:]

Hfile_ptc << (H0,t)
Ufile_ptc << (Us,t)
bfile_ptc << (S0,t)


"""
for dts in dt_schedule:
    dt.assign(dts)
    t += dts

    #mass_solver.solve(l_bound,u_bound)
    solve(R==0, U, bcs=bcs+[bc_2])
    assigner_inv.assign([un,vn,u2n,v2n,H0],U)

    #H0_temp = project(H)
    #H0.vector()[:] = H0_temp.vector()[:]
    
    Us_temp = project(as_vector([u(0),v(0)]))
    Us.vector()[:] = Us_temp.vector()[:]

    S_temp = project(S)
    S0.vector()[:] = S_temp.vector()[:]

    Hfile_ptc << (H0,t)
    Ufile_ptc << (Us,t)
    bfile_ptc << (S0,t)
"""
File(results_dir + 'Ustar.xml') << U 



