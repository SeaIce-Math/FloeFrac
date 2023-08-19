import fenics as fs

"""
Example code generates fractures on a 1km-by-1km square domain using a staggered solver and phase-field formulation the fracture set.
"""

#terminal number of staggered solves and error
solvTol = 5000
errTol = 10**(-6)

"""
phase-field and fracture parameters
ell is the phase-field regularity parameter
ne is the residual elasticity
gc is the fracture toughness
"""
ell = 10**(-4)
ne=10**(-4)
gc=200

"""
displacements on left edge
dispX is the horizontal displacement
dispY is the vertical displacement
"""
dispX = 5
dispY = 0

#Lam\'e parameters
lmbda = 9.33*(10**9)
mu = 3.52*(10**9)

"""
Generates mesh and function spaces.

Can save mesh using code snippet given below.

mesh_file = fs.File("FracDemoMesh.xml")
mesh_file << mesh

"""
mesh = fs.RectangleMesh(fs.Point(0,0),fs.Point(1000,1000),200,200)
V = fs.VectorFunctionSpace(mesh, 'P', degree=1)
ps = fs.FunctionSpace(mesh,"Lagrange",1)
u = fs.TrialFunction(V)
v = fs.TestFunction(V)
s = fs.TrialFunction(ps)
w = fs.TestFunction(ps)

"""
function pairs for the staggered algorithm
u0 and u1 are the vector-valued displacement fields
s0 and s1 are the scalar-valued phase fields
"""
u1 = fs.Function(V)
u0 = fs.Function(V)
s1 = fs.Function(ps)
s0 = fs.Function(ps)

#initial phase-field, unbroken
s0.vector()[:] = 1
s1.vector()[:] = 1

"""
operators and energy fields defined here.
epsln is the strain-rate tensor, a symmetric gradient, that acts on the displacement field.
sigma the stress tensor.
psi is the scalar-valued elastic energy field.
phase is the scalar-valued surface energy field that tracks the energy used to fracture at a point.
"""
def epsln(v):
    return fs.sym(fs.grad(v))

def sigma(v):
    return lmbda*fs.tr(epsln(v))*fs.Identity(2) + 2*mu*epsln(v)

def psi(v):
    return (
        0.5*lmbda*fs.tr(epsln(v))**2 
        + mu*fs.tr(fs.dot(epsln(v),epsln(v)))
        )

def phase(w):
    return gc*(
        (1 - w)**2/(4*ell) 
        + ell*abs(fs.div(fs.grad(w)))
        )

"""
variational problems defined here.
ae and Le are the right- and left-hand sides of the elasticity equation, respectively. Similarly, ap and Lp are the sides of the phase-field equation.
"""

ae = (s0**2 + ne)*fs.inner(sigma(u), fs.grad(v))*fs.dx
Le = fs.dot(fs.Constant((0,0)), v)*fs.ds

ap = (
    s*w*psi(u0)*fs.dx 
    + gc*ell*fs.dot(fs.grad(s),fs.grad(w))*fs.dx 
    + (gc/(4*ell))*s*w*fs.dx
    )
Lp = (gc/(4*ell))*w*fs.dx

"""
boundary conditions are defined here.
displacements dispX and dispY are applied.
the material is unbroken on the boundary, s = 1.
"""

def bdry(x, on_boundary):
    return on_boundary

class BdryRightFixd(fs.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fs.near(x[0],1000)

class BdryLeftDisp(fs.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fs.near(x[0],0)

bcER = fs.DirichletBC(V, (0,0), BdryRightFixd())
bcEL = fs.DirichletBC(V, (dispX,dispY), BdryLeftDisp())
bcE = [bcER,bcEL]
bcP = fs.DirichletBC(ps, fs.Constant(1.0), bdry)

"""
FEniCS solvers are instanced here.
note: Below, the python3 code lists available solvers.

from fenics import *
list_linear_solver_methods()

"""

problemE = fs.LinearVariationalProblem(ae, Le, u1, bcE)
solverE = fs.LinearVariationalSolver(problemE)
solverE.parameters['linear_solver'] = "mumps"

problemP = fs.LinearVariationalProblem(ap, Lp, s1, bcP)
solverP = fs.LinearVariationalSolver(problemP)
solverP.parameters['linear_solver'] = "mumps"

"""
the following function excutes the staggered algorithm. it is called afterwards. 

one may use 'fs.plot(s0)' to print the output which saves a matplotlib.pyplot plot. alternatively, solutions can be saved and viewed in Paraview with the snippet given below.

vtkfile = File('phase.pvd')
vtkfile << s0

may save the numerical values of the profile with following code.

numpy.savetxt('phase',s0.vector())

may load using the following. note: needs to the same mesh and vector space.

s0.vector()[:] = numpy.loadtxt('phase')
"""

def fracAlg(dispX,dispY,prnt=1):
    s0.vector()[:] = 1
    s1.vector()[:] = 1
    n=0
    tol=1
    while n<solvTol and tol > errTol:
        solverE.solve()
        u0.assign(u1)
        solverP.solve()
        tol = fs.norm(s0.vector() - s1.vector(), 'linf')
        s0.assign(s1)
        n += 1
    if prnt == 1:
        print("solves: "+str(n)+", err: " + str(tol))

fracAlg(dispX,dispY)