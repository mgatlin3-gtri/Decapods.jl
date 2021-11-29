# Our developed libraries
using CombinatorialSpaces
using CombinatorialSpaces.ExteriorCalculus

using Catlab.Present
using Catlab.Graphics
using Catlab.CategoricalAlgebra
using LinearAlgebra

using Decapods.Simulations
using Decapods.Examples
using Decapods.Diagrams
using Decapods.Schedules

# Julia community libraries
using MeshIO
using CairoMakie
using Decapods.Debug
using DifferentialEquations
using Logging: global_logger


# Define used quantities
@present EM2DQuantities(FreeExtCalc2D) begin
    X::Space
    neg₁::Hom(Form1(X), Form1(X)) # -1
    E::Hom(munit(), Form1(X))     # electric field
    B::Hom(munit(), Form2(X))     # magnetic field
    c::Hom(Form2(X), Form2(X))   # μ₀ / ϵ₀ (scalar)
end

# Define Electromagnetic physics
@present EM2D <: EM2DQuantities begin
    B ⋅ ∂ₜ(Form2(X)) == E ⋅ neg₁ ⋅ d₁(X)
    E ⋅ ∂ₜ(Form1(X)) == B ⋅ c ⋅ ⋆₂(X) ⋅ dual_d₀(X) ⋅ ⋆₁⁻¹(X)
end

diag = eq_to_diagrams(EM2D)
to_graphviz(diag; edge_len = "1.3")
##

dwd = diag2dwd(diag)
to_graphviz(dwd, orientation = LeftToRight)
##

s = EmbeddedDeltaSet2D("../meshes/disk_0_25.stl")
sd = dual(s);

# Define non-default operators (multiplication by constants)
c = -1 * (1.2566e-6 / 8.8542e-12)
# c = -1 * (1.2566 / 8.8542)
funcs = sym2func(sd)


funcs[:c] = Dict(:operator => c * I(ntriangles(s)), :type => MatrixFunc())
funcs[:neg₁] = Dict(:operator => -1 * I(ne(sd)), :type => MatrixFunc())

# Define initial conditions
x = [p[1] for p in s[:point]];

# # sinusoidal distribution for E-field
# eField(x) = begin
#     amp = sin((3) * π * (x[1]))
#     amp * Point{3,Float32}(1, 0, 0)
# end

# radial distribution for E-field
r = 10
eField(x) = begin
    amp = (4 * pi * r^2) * x[1] * x[1]
    amp * Point{3,Float32}(1, 0, 0)
end

E = ♭(sd, DualVectorField(eField.(sd[triangle_center(sd), :dual_point])))

B = TriForm(zeros(ntriangles(s)))

u0 = vcat(B.data, E.data) # must be in order of presentation

# Compress wiring diagram over contiguous matrix multiplications
cont_dwd = deepcopy(dwd)
Examples.contract_matrices!(cont_dwd, funcs)

# Generate simulation function
func, _ = gen_sim(cont_dwd, funcs, sd; autodiff = false);

# Solve problem
prob = ODEProblem(func, u0, (0.0, 0.5));
sol = solve(prob, Tsit5());


# Key for debugging simulation
sim_key(dwd, orientation = LeftToRight)


exp_func, _ = gen_sim(dwd, funcs, sd; autodiff = false);

# 
fig, ax, ob = draw_wire(s, sd, dwd, exp_func, sol[10], 6)
fig


# Plot solution
B_range = 1:ntriangles(s)

fig, ax, ob = draw_wire(s, sd, dwd, exp_func, sol[10], 3; colorrange = (-1e-1, 1e-1))
fig


# Record solution
times = range(1e-4, sol.t[end], length = 500)
colors = [vcat([[v, v, v] for v in sol(t)[B_range]]...) for t in times]


framerate = 30

record(fig, "magnetic_field-radialDist_025.gif", collect(1:length(collect(times))); framerate = framerate) do i
    ob.color = colors[i]
    ob.colorrange = (-1e-1, 1e-1)
end