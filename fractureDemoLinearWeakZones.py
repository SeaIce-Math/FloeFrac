import fenics as fs
import linFieldGen as lfg

"""
Example code generates fractures on a 1km-by-1km square domain using a staggered solver and phase-field formulation the fracture set. Random linear weakness are inserted. Starting positions and angles are sampled uniformly. Lines are mollified. The number of lines if currently fixed.

Note: comments redundant with "fractureDemo.py" have been removed. Uses methods from "linFieldGen" module.
"""

solvTol = 5000
errTol = 10**(-6)

ell = 10**(-4)
ne=10**(-4)
gc=200

dispX = 5
dispY = 0

"""
Lame-Field parameters

lmbdaU is the unweakened value for lambda. 
muU and mu correspond similarly.
lmeContrst is the contrast between the weakened unweakened regions.
nLines is the number of sampled lines.
"""

lmbdaU = 9.33*(10**9)
muU = 3.52*(10**9)
lmeContrst = 1/10
nLines = 3

mesh = fs.RectangleMesh(fs.Point(0,0),fs.Point(1000,1000),200,200)
V = fs.VectorFunctionSpace(mesh, 'P', degree=1)
ps = fs.FunctionSpace(mesh,"Lagrange",1)
u = fs.TrialFunction(V)
v = fs.TestFunction(V)
s = fs.TrialFunction(ps)
w = fs.TestFunction(ps)

u1 = fs.Function(V)
u0 = fs.Function(V)
s1 = fs.Function(ps)
s0 = fs.Function(ps)

s0.vector()[:] = 1
s1.vector()[:] = 1

"""
Lame fields sampled here. Uses methods from "linFieldGen" module.

Next six lines of code sample the lines and create an object that contains (as an attribute) the mollified indicator function of a set of lines.

Following the mollified function is interpolated onto the FEM space of the phase-field. MollInterp is the interpolated function.

Operator are defined with the lame fields.
"""

dLnFld = lfg.DiscreteLinearField(10)
dLnFld.samplePointsUnif(0,1000,0,1000)
dLnFld.sampleAnglesUnif()
dLnFld.sampleLengthsUnif(0, 1500)
dLnFld.sampleContrastsUnif(0.1,0.9)
dLnFld.linearField2MollifiedIndicator(5,5)

class aFunc(fs.UserExpression):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func
    def eval(self, values, x):
        values[0] = self.func(x[0],x[1])
    def value_shape(self):
        return ()

mollFExp = aFunc(func=dLnFld.mollifiedIndicator)
mollInterp = fs.interpolate(mollFExp,ps)

def epsln(v):
    return fs.sym(fs.grad(v))

def sigma(v):
    return (
        (lmbdaU*mollInterp)*fs.tr(epsln(v))*fs.Identity(2) 
        + 2*(muU*mollInterp)*epsln(v)
        )

def psi(v):
    return (
        0.5*(lmbdaU*mollInterp)*fs.tr(epsln(v))**2 
        + (muU*mollInterp)*fs.tr(fs.dot(epsln(v),epsln(v)))
        )

def phase(w):
    return gc*(
        (1 - w)**2/(4*ell) 
        + ell*abs(fs.div(fs.grad(w)))
        )

ae = (s0**2 + ne)*fs.inner(sigma(u), fs.grad(v))*fs.dx
Le = fs.dot(fs.Constant((0,0)), v)*fs.ds

ap = (
    s*w*psi(u0)*fs.dx 
    + gc*ell*fs.dot(fs.grad(s),fs.grad(w))*fs.dx 
    + (gc/(4*ell))*s*w*fs.dx
    )
Lp = (gc/(4*ell))*w*fs.dx

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

problemE = fs.LinearVariationalProblem(ae, Le, u1, bcE)
solverE = fs.LinearVariationalSolver(problemE)
solverE.parameters['linear_solver'] = "mumps"

problemP = fs.LinearVariationalProblem(ap, Lp, s1, bcP)
solverP = fs.LinearVariationalSolver(problemP)
solverP.parameters['linear_solver'] = "mumps"

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