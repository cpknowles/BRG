from cslvr   import *

"""
  Things to Do:

  insert boundary conditions that velocity goes to 0 at boundaries doesn't seem to try to calc that currently

  start time steping so I can start trying to calculate changes in thickness

  Obtain correct surface for flow to occur
    split into subdomains to allow mainflow to have different slope from Loket?
"""

#######################################################################
###################Mesh Creation, start D2model #######################
#######################################################################

#Read in Mesh from gmsh .xml file
mesh1    = Mesh("2dmesh.xml")

coor = mesh1.coordinates()
boundary_parts = MeshFunction('size_t', mesh1, mesh1.topology().dim()-1)
#directory for results
plt_dir = './velbalance_results/'

# the balance velocity uses a 2D-model :
model = D2Model(mesh1, out_dir = 'plt_dir', order=1)

V = VectorFunctionSpace(mesh1, "Lagrange", 2)
Q = FunctionSpace(mesh1, "Lagrange", 1)
W = V * Q


# Variables and their meanings
"""
    Q      = model.Q      function space?
    S      = model.S      Surface of glacier, dictates flow of glacier
    B      = model.B      bed of glacier, set to 0 as flow depends on surface
    H      = S - B        Ice thickness
    h      = model.h       
    N      = model.n       glenn flow exponent
    uhat   = model.uhat    uhat = model.vert_integrate(u, d='up')
    vhat   = model.vhat     normalized flux direction from \nabla S
    adot   = model.adot    accumulation/ablation function 
    Fb     = model.Fb      basal-water discharge
"""

######################################################################
####################### No Slip Boundary #############################
######################################################################

class MainBound(SubDomain):
  def inside(model, x, on_boundary):
    return on_boundary and \
      (x[0] > 2.0 - DOLFIN_EPS or \
      (x[0] < 1.0 + DOLFIN_EPS and x[1] > -1.0 -DOLFIN_EPS) or  \
      (x[0] < 1.0 + DOLFIN_EPS and x[1] < -2.0 + DOLFIN_EPS ))  #Last Operator was >, I think it should be less than the lower bound of Loket
mainbc = MainBound()
mainbc.mark(boundary_parts,0)

class LoketBound(SubDomain):
  def inside(model, x, on_boundary):
    return on_boundary and ((x[0] > 0.0 - DOLFIN_EPS and x[0] < 1.0 + DOLFIN_EPS and \
      x[1] > -1.0- DOLFIN_EPS) or (x[0] > 0.0 - DOLFIN_EPS and x[0] < 1.0 +DOLFIN_EPS and \
      x[1] < -2.0 + DOLFIN_EPS))
loketbc = LoketBound()
loketbc.mark(boundary_parts,1)

BCS = [DirichletBC(W.sub(0), (0,0), boundary_parts, 0),
     DirichletBC(W.sub(0), (0,0), boundary_parts, 1)]


#######################################################################
##########################Calc S and B, plot S ########################
#######################################################################
S0 = 400   #Surface of Glacier at Acculumation zone
B0 = 0     #Bed of Galcier, doesn't affect flow
L= 4       #Length of Domain, set to give gradient of glacier slope within mesh


#Hardcoded Surface and Bed values, currently set to be sloped so we have flow down tributary and main flow
#Degree 3 to let me have (x,y,z)
#CHANGE THIS TO BE IN TERMS OF ANGLE!!!!
S = Expression('S0 + S0*x[1]/(.8*L)  - S0*x[0]*x[0]/(2*L)', S0=S0,L=L, degree=3)
B = Expression('B0', B0=B0, degree=3)


#Inintialize my bed and surface expressions in the model
model.init_B(B)
model.init_S(S)
#Supposed to initialize Glenn's Flow exponent, doesn't change output
model.n(3)

#Figure out change in Surface, plot surface with countours for height
S_min = model.S.vector ().min() 
S_max = model.S.vector ().max() 
S_lvls = array([S_min, 200 , 300 , 400, 500, S_max])

#Plot Surface
plot_variable(u = model.S, name = 'S', direc = plt_dir , figsize = (5,5), levels = S_lvls , tp = True , show = False , cb = False , contour_type = 'lines', hide_ax_tick_labels = True)


#Won't Graph without this on. I should put in old BCS from Blackrapids.py????
model.calculate_boundaries(latmesh=False, mask=None, lat_mask=None, adot=None, U_mask=None, mark_divide=False)


#######################################################################
################### Solve BalanceVelocity and Graph ###################
#######################################################################

#direction of flow used in BalanceVelocity.direction_of_flow()
d = (model.S.dx(0),-model.S.dx(1))

#kappa is floating-point value representing surface smoothing radius in units of ice thickness 'H = S-B'
#kappas  = [0,5,10]
#methods = ['SUPG', 'SSM', 'GLS']
kappa = 10
methods = 'GLS'


#######################################################################
################### Initial Balancevelocity         ###################
#######################################################################

S      = model.S
B      = model.B
H      = S - B
h      = model.h
N      = model.N
uhat   = model.uhat
vhat   = model.vhat
adot   = model.adot
Fb     = model.Fb
 # form to calculate direction of flow (down driving stress gradient) :
phi   = TestFunction(Q)
ubar  = TrialFunction(Q)
kappa = Constant(kappa)
    
    # stabilization test space :
Uhat     = as_vector([uhat, vhat])
tau      = 1 / (2*H/h + div(H*Uhat))
phihat   = phi + tau * dot(Uhat, grad(phi)) 
   
# the left-hand side : 
def L(u):      return u*H*div(Uhat) + dot(grad(u*H), Uhat)
def L_star(u): return u*H*div(Uhat) - dot(grad(u*H), Uhat)
def L_adv(u):  return dot(grad(u*H), Uhat)
   
Nb = sqrt(B.dx(0)**2 + B.dx(1)**2 + 1) 
Ns = sqrt(S.dx(0)**2 + S.dx(1)**2 + 1)
f  = Ns*adot - Nb*Fb

# use streamline-upwind/Petrov-Galerkin :
if stabilization_method == 'SUPG':
      s      = "    - using streamline-upwind/Petrov-Galerkin stabilization -"
      model.B = + L(ubar) * phi * dx \
               + inner(L_adv(phi), tau*L(ubar)) * dx
      model.a = + f * phi * dx \
               + inner(L_adv(phi), tau*f) * dx

    # use Galerkin/least-squares
elif stabilization_method == 'GLS':
      s      = "    - using Galerkin/least-squares stabilization -"
      model.B = + L(ubar) * phi * dx \
               + inner(L(phi), tau*L(ubar)) * dx
      model.a = + f * phi * dx \
               + inner(L(phi), tau*f) * dx

    # use subgrid-scale-model :
elif stabilization_method == 'SSM':
      s      = "    - using subgrid-scale-model stabilization -"
      model.B = + L(ubar) * phi * dx \
               - inner(L_star(phi), tau*L(ubar)) * dx
      model.a = + f * phi * dx \
               - inner(L_star(phi), tau*f) * dx
    
print_text(s, cls=model)



#######################################################################
################### Solve direction of flow         ###################
#######################################################################

phi   = TestFunction(Q)
d_x   = TrialFunction(Q)
d_y   = TrialFunction(Q)
kappa = Constant(model.kappa)
    
# horizontally smoothed direction of flow :
a_dSdx = + d_x * phi * dx + (kappa*H)**2 * dot(grad(phi), grad(d_x)) * dx \
         - (kappa*H)**2 * dot(grad(d_x), N) * phi * ds
L_dSdx = d[0] * phi * dx
    
a_dSdy = + d_y * phi * dx + (kappa*H)**2 * dot(grad(phi), grad(d_y)) * dx \
         - (kappa*H)**2 * dot(grad(d_y), N) * phi * ds
L_dSdy = d[1] * phi*dx
    
# update velocity direction :
s    = "::: solving for smoothed x-component of flow direction " + \
           "with kappa = %g :::" % model.kappa
print_text(s, cls=model)
solve(a_dSdx == L_dSdx, model.d_x, annotate=annotate)
print_min_max(model.d_x, 'd_x')
    
s    = "::: solving for smoothed y-component of flow direction " + \
           "with kappa = %g :::" % model.kappa
print_text(s, cls=model)
solve(a_dSdy == L_dSdy, model.d_y, annotate=annotate)
print_min_max(model.d_y, 'd_y')
    
# normalize the direction vector :
s    =  r"::: calculating normalized flux direction from \nabla S:::"
print_text(s, cls=model)
d_x_v = model.d_x.vector().array()
d_y_v = model.d_y.vector().array()
d_n_v = np.sqrt(d_x_v**2 + d_y_v**2 + 1e-16)
model.assign_variable(model.uhat, d_x_v / d_n_v)
model.assign_variable(model.vhat, d_y_v / d_n_v)


#######################################################################
############################ Solve  ###################################
#######################################################################

s    = "::: solving velocity balance magnitude :::"
print_text(s, cls=model)
solve(model.B == model.a, model.Ubar, annotate=annotate)
print_min_max(model.Ubar, 'Ubar')
    
# enforce positivity of balance-velocity :
s    = "::: removing negative values of balance velocity :::"
print_text(s, cls=model)
Ubar_v = model.Ubar.vector().array()
Ubar_v[Ubar_v < 0] = 0
model.assign_variable(model.Ubar, Ubar_v)



#######################################################################
############################ Graph  ###################################
#######################################################################


U_max  = model.Ubar.vector().max()
U_min  = model.Ubar.vector().min()
#U_lvls = array([U_min, 2, 10, 20, 50, 100, 200, 500, 1000, U_max])
#hand chose intervals to have contours on graph, will need to generalize later
U_lvls = array([U_min,.4,.8,1.2,1.4,1.5,1.7,2,2.4,2.8,3.2,3.6,U_max])
name = 'Ubar_%iH_kappa_%i_%s' % (5, kappa, method)
tit  = r'$\bar{u}_{%i}$' % kappa
#plot_variable(model.Ubar, name=name, direc=plt_dir,
             # figsize             = (8,3),
             # equal_axes          = False)

plot_variable(model.Ubar , name=name , direc=plt_dir ,
              title=tit ,  show=False ,
              levels=U_lvls , tp=False , cb_format='%.1e')






"""
for kappa in kappas:

  for method in methods:

    bv = BalanceVelocity(model, kappa=kappa, stabilization_method=method)
    bv.solve_direction_of_flow(d)
    bv.solve()

    U_max  = model.Ubar.vector().max()
    U_min  = model.Ubar.vector().min()
    #U_lvls = array([U_min, 2, 10, 20, 50, 100, 200, 500, 1000, U_max])
    #hand chose intervals to have contours on graph, will need to generalize later
    U_lvls = array([U_min,.4,.8,1.2,1.4,1.5,1.7,2,2.4,2.8,3.2,3.6,U_max])
    name = 'Ubar_%iH_kappa_%i_%s' % (5, kappa, method)
    tit  = r'$\bar{u}_{%i}$' % kappa
    #plot_variable(model.Ubar, name=name, direc=plt_dir,
             # figsize             = (8,3),
             # equal_axes          = False)

    plot_variable(model.Ubar , name=name , direc=plt_dir ,
                  title=tit ,  show=False ,
                  levels=U_lvls , tp=False , cb_format='%.1e')
"""
