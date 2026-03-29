# Euler Flow Solver (G2 Stabilized)

This repository contains a high-performance computational fluid dynamics (CFD) solver based on the **FEniCSx** (Dolfinx) library. It is designed to solve the **Incompressible Euler Equations** using a General Galerkin (G2) stabilization method, specifically applied to external aerodynamics over an F-35 airframe.

---

## Governing Equations

The solver addresses the Euler equations for an inviscid, incompressible fluid. The primary variables are the velocity vector $u$ and the scalar pressure $p$.

### 1. Momentum and Continuity
The strong form of the equations is:

$$
\frac{\partial u}{\partial t} + (u \cdot \nabla) u + \nabla p = 0
$$

$$
\nabla \cdot u = 0
$$

### 2. G2 Stabilization (General Galerkin)
To handle the instabilities inherent in high-speed, low-viscosity flows, this solver implements a **G2 stabilization**. The residual of the momentum equation is defined as:

$$
\mathcal{R}_m = \frac{u - u_{old}}{\Delta t} + (u_{old} \cdot \nabla) u + \nabla p
$$

The stabilization parameter $\delta$ is calculated locally based on the mesh diameter $h$ and the time step $\Delta t$:

$$
\delta = \frac{h}{2 \sqrt{\|u_{old}\|^2 + (h/\Delta t)^2}}
$$

### 3. Variational Formulation
The stabilized weak form $F$ solved at each time step is:

$$
F = \int_\Omega \left( \left( \frac{u - u_{old}}{\Delta t} + (u_{old} \cdot \nabla) u \right) \cdot v - p(\nabla \cdot v) + q(\nabla \cdot u) \right) dx + \int_\Omega \delta \mathcal{R}_m \cdot \left( (u_{old} \cdot \nabla) v + \nabla q \right) dx
$$

---

## Implementation Details

* **Space Discretization:** $cG(1)cG(1)$ (Continuous Galerkin) for both velocity and pressure using a mixed element space.
* **Time Integration:** Implicit Euler scheme with a time step of $\Delta t = 0.005$.
* **Linear Solver:** Uses **PETSc** with the **MUMPS** direct solver (LU decomposition) for robust convergence.
* **Boundary Conditions:**
    * **Inlet:** Dirichlet condition ($U_\infty = 10.0$) at the $x_{min}$ boundary.
    * **Surface:** Nitsche-style slip condition applied to the aircraft facets using a penalty parameter $\gamma = 10.0$.

---

## Usage

### Prerequisites
You need a working installation of `dolfinx`, `mpi4py`, `ufl`, and `petsc4py`. 

### Required Files
The solver expects the following XDMF mesh files:
* `f35_domain_lvl2.xdmf` (Volume mesh)
* `f35_facets_lvl2.xdmf` (Boundary facets)

### Running the Solver
To run the solver in parallel using MPI:
```bash
mpirun -n 4 python3 solver_euler.py
