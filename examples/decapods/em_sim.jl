# Our developed libraries
using CombinatorialSpaces
using CombinatorialSpaces.ExteriorCalculus


using Catlab.Present
using Catlab.Graphics
using Catlab.CategoricalAlgebra
using LinearAlgebra

# Julia community libraries
using MeshIO
using CairoMakie
using DifferentialEquations
using Distributions
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

#using Revise
using Decapods.Simulations
using Decapods.Examples
using Decapods.Diagrams
using Decapods.Schedules
using Decapods.Debug

# Define used quantities
@present EM2DQuantities(FreeExtCalc2D) begin
    X::Space
    E::Hom(munit(), Form1(X))     # electric field
    B::Hom(munit(), DualForm0(X))     # magnetic field
    c₀::Hom(Form2(X), Form2(X))   # 1/(μ₀ ϵ₀) (scalar)
end

# Define Electromagnetic physics
@present EM2D <: EM2DQuantities begin
    B ⋅ ∂ₜ(DualForm0(X)) == E ⋅ d₁(X) ⋅ c₀ ⋅ ⋆₂(X)
    E ⋅ ∂ₜ(Form1(X)) == B ⋅ dual_d₀(X) ⋅ ⋆₁⁻¹(X)
end

diag = eq_to_diagrams(EM2D)
to_graphviz(diag; edge_len = "1.3")

dwd = diag2dwd(diag)
to_graphviz(dwd, orientation = LeftToRight)

using SpecialFunctions

s = EmbeddedDeltaSet2D("../meshes/disk_0_50.stl")
#s = EmbeddedDeltaSet2D{Bool, Point{3, Float64}}()
#copy_parts!(s, s1)
orient_component!(s, 1, false)
#s = EmbeddedDeltaSet2D("../../meshes/pipe_fine.stl")
s[:point] .= s[:point] ./ 10
sd = dual(s, subdiv = DiscreteExteriorCalculus.Barycenter());

c₀ = 1

funcs = sym2func(sd)
funcs[:c₀] = Dict(:operator => c₀ * I(ntriangles(s)), :type => MatrixFunc())
funcs[:neg₁] = Dict(:operator => -1 * I(ne(sd)), :type => MatrixFunc())
funcs[:⋆₁⁻¹] = Dict(:operator => inv_hodge_star(Val{1}, sd, hodge = DiagonalHodge()), :type => MatrixFunc())


# Exact solutions for E and B (m = order of Bessel J function, n = order of Bessel Y function ω₀ = frequency, ϕ = phase, a_star = 1, c = speed, I₀ = current)

exactE(x, y, t; ω₀ = 0, ϕ = 0, a_star = 1, c = 0, I₀ = 0.1) =
    [(-1 * pi / 2) * I₀ * ω₀ * a_star * bessely0(ω₀ * a_star) * besselj0(ω₀ * sqrt(x^2 + y^2) / c) * sin(ω₀ * t + ϕ)] -
    [(1 * pi / 2) * I₀ * ω₀ * a_star * bessely0(ω₀ * a_star) * besselj0(ω₀ * sqrt(x^2 + y^2) / c) * cos(ω₀ * t + ϕ)]

exactB(x, y, t; ω₀ = 0, ϕ = 0, a_star = 1, c = 0, I₀ = 0.1) =
    [1 / c * (-1 * pi / 2) * I₀ * ω₀ * a_star * bessely0(ω₀ * a_star) * besselj0(ω₀ * sqrt(x^2 + y^2) / c) * cos(ω₀ * t + ϕ)] +
    [1 / c * (1 * pi / 2) * I₀ * ω₀ * a_star * bessely0(ω₀ * a_star) * besselj0(ω₀ * sqrt(x^2 + y^2) / c) * sin(ω₀ * t + ϕ)]


# Initial conditions

eField(v) = begin
    r = sqrt(v[1]^2 + v[2]^2)
    mag = exactE(v[1], v[2], 0; ω₀ = 5.5201, ϕ = 0, c = c₀)
    x = -1 * v[2] / r
    y = v[1] / r
    mag[1] * Point{3,Float32}(x, y, 0.0)
end

bField(z) = begin
    mag = exactB(z[1], z[2], 0; ω₀ = 5.5201, ϕ = 0, c = c₀)
    mag[1]
end

EFd = ♭(sd, DualVectorField(eField.(sd[triangle_center(sd), :dual_point]))).data
EF = zeros(ne(s))

B = [bField(sd[sd[t, :tri_center], :dual_point]) for t = 1:ntriangles(s)]
u0 = vcat(B, EF)

cont_dwd = deepcopy(dwd)
Examples.contract_matrices!(cont_dwd, funcs)

f1, f2, dwd1, dwd2 = gen_leapfrog(diag, funcs, sd, [:B, :E]);
tspan = (0.0, 2.0)
dyn_prob = DynamicalODEProblem(f1, f2, B, EF, tspan)
sol = solve(dyn_prob, VerletLeapfrog(); dt = 0.001, progress = true, progress_steps = 100);

to_graphviz(dwd2, orientation = LeftToRight)

# Key for debugging simulation
# sim_key(dwd, orientation = LeftToRight)

exp_func, _ = gen_sim(dwd, funcs, sd; autodiff = false);

fig, ax, ob = draw_wire(s, sd, dwd, exp_func, sol(0.1), 3)
ob.color = vcat([[v, v, v] for v in sol(2.0)[1:ntriangles(s)]]...)
fig
save("em_res.png", fig)

# Record exact solution
times = range(1e-4, sol.t[end], length = 300)
colors = [vcat([[v, v, v] for v in [exactB(v[1], v[2], t; ω₀ = 5.5201, ϕ = 0, c = c₀)[1] for v in sd[sd[:tri_center], :dual_point]]]...) for t in times]
colorrange = maximum(vcat(colors...))

framerate = 30

record(fig, "magnetic_field-disk-exact.gif", collect(1:length(collect(times))); framerate = framerate) do i
    ob.color = colors[i]
    ob.colorrange = (-0.3, 0.3)
end

# Record simulated solution
times = range(1e-4, sol.t[end], length = 300)
colors = [vcat([[v, v, v] for v in sol(t)[1:ntriangles(s)]]...) for t in times]
colorrange = maximum(vcat(colors...))

framerate = 30

record(fig, "magnetic_field-disk.gif", collect(1:length(collect(times))); framerate = framerate) do i
    ob.color = colors[i]
    ob.colorrange = (-0.3, 0.3)
end

# Plot the difference between the exact and simulated results

# todo: relative = |exact - sim|/|exact| and absolute error: |exact - sim|

# absolute:
times = range(1e-4, sol.t[end], length = 300)
errors = [sol(t)[1:ntriangles(s)] .- [exactB(p[1], p[2], t; ω₀ = 5.5201, ϕ = 0, c = c₀)[1] for p in sd[sd[:tri_center], :dual_point]] for t in times]
print(errors[1])

r_max = maximum(abs.(errors[end]))
fig, ax, ob = draw_wire(s, sd, dwd, exp_func, sol(0.1), 3)
ob.color .= [vcat([[v, v, v] for v in errors[i]]...) for i in errors]
# ob.color = [vcat([[v, v, v] for v in errors[i]]...) for i in 1:length(errors)]
ob.colorrange = (-r_max, r_max)
ob.colormap = :bluesreds
fig

r_max = maximum(abs.(errors[end]))
colors = [vcat([[v, v, v] for v in errors[i]]...)]
framerate = 30
record(fig, "error_EM.gif", collect(1:length(collect(times))); framerate = framerate) do i
    ob.color = colors[i]
    ob.colorrange = (-r_max, r_max)
    ob.colormap = :bluesreds
end