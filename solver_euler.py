# =====================
# 0 - IMPORT LIBRARIES
# =====================

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
import basix.ufl 

# ==============================================
# 1 - LOAD MESHES ( FROM VSPAIRSHOW OR WHATEVS )
# ==============================================

mesh_comm = MPI.COMM_WORLD

# READ VOLUME

with io.XDMFFile(mesh_comm, "f35_domain_lvl2.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

# READ SURFACE

with io.XDMFFile(mesh_comm, "f35_facets_lvl2.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="Grid")

# ================================
# 2 - SPACE FUNCTIONS (cG(1)cG(1))
# ================================

v_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
p_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
W = fem.functionspace(domain, basix.ufl.mixed_element([v_el, p_el]))

# ========================
# 3 - BOUNDARY CONDITIONS
# ========================

V, _ = W.sub(0).collapse()

# INITIAL WIND VZLOCITY

U_inf = 10.0
u_inlet_val = fem.Function(V)
u_inlet_val.interpolate(lambda x: np.vstack((np.full(x.shape[1], U_inf), 
                                             np.zeros(x.shape[1]), 
                                             np.zeros(x.shape[1]))))

# LOCALIZE OUTLET (X MIN)

x_min_all = domain.comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN)
def inlet_marker(x):
    return np.isclose(x[0], x_min_all, atol=1e-2)

inlet_dofs = fem.locate_dofs_geometrical((W.sub(0), V), inlet_marker)
bc_inlet = fem.dirichletbc(u_inlet_val, inlet_dofs, W.sub(0))

bcs = [bc_inlet] 

# ==========================================
# 4 - VARIATIONAL FORMULATION G2 ( EULER )
# ==========================================

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

w_old = fem.Function(W)
u_old, p_old = ufl.split(w_old)

dt = fem.Constant(domain, default_scalar_type(0.005)) 
h = ufl.CellDiameter(domain)

# MOMENTUM RESIDUAL

res_m = (u - u_old)/dt + ufl.dot(u_old, ufl.grad(u)) + ufl.grad(p)
delta = h / (2.0 * ufl.sqrt(ufl.inner(u_old, u_old) + (h/dt)**2))

# G2 FORM
F = (ufl.inner((u - u_old)/dt + ufl.dot(u_old, ufl.grad(u)), v) - p * ufl.div(v) + q * ufl.div(u)) * ufl.dx
F += delta * ufl.inner(res_m, ufl.dot(u_old, ufl.grad(v)) + ufl.grad(q)) * ufl.dx

# SLIP CONDITION

n = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
gamma = fem.Constant(domain, default_scalar_type(10.0)) 

F += (gamma / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(2) 

a = ufl.lhs(F)
L = ufl.rhs(F)

# =====================
# 5 - SOLVER TIME LOOP
# =====================

a_form = fem.form(a)
L_form = fem.form(L)
w_h = fem.Function(W)

solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps") 

t = 0.0
T_final = 2.0 # 2 SECONDS ON THE CLUSER MAYBEEE
passo = 0

print("Iniciando simulação do F-35...")
while t < T_final:
    t += dt.value
    passo += 1
    
    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    solver.setOperators(A)
    
    b = assemble_vector(L_form)
    fem.petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)
    
    solver.solve(b, w_h.x.petsc_vec)
    w_old.x.array[:] = w_h.x.array[:]
    
    if passo % 10 == 0:
        print(f"Tempo: {t:.3f}s concluído.")

# ==========================================
# 6- SAVING RESULTS FILE (results_f35.xdmf)
# ==========================================

u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()
p_h.name = "Pressao"
u_h.name = "Velocidade"

with io.XDMFFile(domain.comm, "results_f35.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_h)
    xdmf.write_function(p_h)

print("SIMULATION FINISHED")