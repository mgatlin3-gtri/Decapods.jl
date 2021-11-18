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
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


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

s = EmbeddedDeltaSet2D("../meshes/wire.obj")
sd = dual(s);

# Define non-default operators (multiplication by constants)
c = -1 * (1.2566e-6 / 8.8542e-12)
funcs = sym2func(sd)

funcs[:c] = Dict(:operator => c * I(ntriangles(s)), :type => MatrixFunc())
funcs[:neg₁] = Dict(:operator => -1 * I(ne(sd)), :type => MatrixFunc())

# Define initial conditions
x = [p[1] for p in s[:point]];

eField(x) = begin
    amp = sin((3) * π * (x[1]))
    amp * Point{3,Float32}(1, 0, 0)
end

E = ♭(sd, DualVectorField(eField.(sd[triangle_center(sd), :dual_point])))

B = TriForm(zeros(ntriangles(s)))

u0 = vcat(B.data, E.data)

# Compress wiring diagram over contiguous matrix multiplications
cont_dwd = deepcopy(dwd)
Examples.contract_matrices!(cont_dwd, funcs)

# Generate simulation function
func, _ = gen_sim(cont_dwd, funcs, sd; autodiff = false);

# # Solve problem
prob = ODEProblem(func, u0, (0.0, 5.0));
sol = solve(prob, Tsit5());


# Key for debugging simulation
sim_key(dwd, orientation = LeftToRight)
##

exp_func, _ = gen_sim(dwd, funcs, sd; autodiff = false);

get_wire(dwd, exp_func, sol(0.0), 3)
##
# Show the flow of concentration as arrows
fig, ax, ob = draw_wire(s, sd, dwd, exp_func, sol[2], 4)
fig
# save("debug_EM.svg")

# # Plot solution
t = 2
sd2 = deepcopy(sd)
new_points = [Point3{Float32}(p[1], 0, sol(t)[B[i]]) for (i, p) in enumerate(sd[triangle_center(sd), :dual_point])]
sd2[triangle_center(sd2), :dual_point] .= new_points

limits = FRect3D(Vec3f0(-1.0, -1.0, -1.0), Vec3f0(1.5, 0.5, 0.6))
fig, ax, ob = wireframe(sd2, limits = limits)

# # Record solution
# tn = Node(0.0)
# sd2 = deepcopy(sd)
# f(t) = begin
#     new_points = [Point3{Float32}(p[1],0,sol(t)[B_range[i]]) for (i,p) in enumerate(sd[triangle_center(sd),:dual_point])] 
#     sd2[triangle_center(sd2),:dual_point] .= new_points
#     sd2
# end

# limits = FRect3D(Vec3f0(-1.0, -1.0, -1.0), Vec3f0(1.5,0.5,0.6))
# fig, ax, ob = wireframe(lift(t->f(t), tn); limits=limits)
# i_range = range(0,10.0, length=100)
# framerate = 30
# record(fig, "EM_3D.gif", i_range; framerate = framerate) do i
#   tn[] = i
# end