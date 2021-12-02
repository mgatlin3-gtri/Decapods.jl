# Our developed libraries
using CombinatorialSpaces
using CombinatorialSpaces.ExteriorCalculus

using Catlab.Present
using Catlab.Graphics
using Catlab.CategoricalAlgebra
using LinearAlgebra
using SpecialFunctions

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

dwd = diag2dwd(diag)
to_graphviz(dwd, orientation = LeftToRight)

# s = EmbeddedDeltaSet2D("../meshes/pipe_fine.stl")
s = EmbeddedDeltaSet2D("../meshes/disk_1_0.stl")
s[:point] .= s[:point] ./ 10
sd = dual(s);

# Define non-default operators (multiplication by constants)
# c = -1 * (1.2566e-6 / 8.8542e-12)
c = -3e8

funcs = sym2func(sd)
funcs[:c] = Dict(:operator => c * I(ntriangles(s)), :type => MatrixFunc())
funcs[:neg₁] = Dict(:operator => -1 * I(ne(sd)), :type => MatrixFunc())


# Bessel functions

exactE(x, y, t; m = 0, ω₀ = 0, A = 1, B = 0, c = sqrt(10), C = 1, D = 0) =
    besselj(m, ω₀ * sqrt(x^2 + y^2)) *
    (A * cos(c * λ[n, m+1] * t) + B * sin(c * λ[n, m+1] * t)) *
    (C * cos((m) * atan(x, y)) + D * sin((m) * atan(x, y)))

exactB(x, y, t;)


y = [exact(p[1], p[2], 0; m = 0, n = 2, A = 5, C = 1) for p in s[:point]];
v = [0.0 for p in s[:point]];

# θ = [atan(p[2] / p[1]) for p in s[:point]]
# r = [sqrt(p[1]^2 + p[2]^2) for p in s[:point]]

# Initial conditions

I₀ = 1
a_star = 1
ω₀ = 5.5201
ϕ = pi / 2

eField(x) = begin
    r = sqrt(x[1]^2 + x[2]^2)
    r_star = r / c
    mag = [(-1 * μ₀ * c * pi / 2) * I₀ * ω₀ * a_star * Y₁ * J₁ * sin(ϕ)]
    -[(μ₀ * c * pi / 2) * I₀ * ω₀ * a_star * J₁ * J₁ * cos(ϕ)]
    mag * Point{3,Float32}(-y, x, 0)
end

EF = ♭(sd, DualVectorField(eField.(sd[triangle_center(sd), :dual_point])))
B = TriForm(zeros(ntriangles(s)))

plot(first.(s[s[:src], :point]), EF.data)


# Generate leapfrog simulation
f1, f2, dwd1, dwd2 = gen_leapfrog(diag, funcs, sd, [:B, :E]);
tspan = (0.0, 10.0)
dyn_prob = DynamicalODEProblem(f1, f2, B, EF, tspan)
sol = solve(dyn_prob, VerletLeapfrog(); dt = 0.01, progress = true, progress_steps = 100);

exp_func, _ = gen_sim(dwd, funcs, sd; autodiff = false);


fig, ax, ob = draw_wire(s, sd, dwd, exp_func, sol[end-1], 6)
fig


# Plot solution
B_range = 1:ntriangles(s)
E_range = 1:ne(s)

fig, ax, ob = draw_wire(s, sd, dwd, exp_func, sol(10.0), 3) #; colorrange = (-1e3, 1e3))
ax.aspect = AxisAspect(1.0)
fig


# For visualizing E-field
# colors = [sol(t)[E_range] for t in times]

# Record solution
times = range(1e-4, sol.t[end], length = 300)
colors = [vcat([[v, v, v] for v in sol(t)[B_range]]...) for t in times]
colorrange = maximum(vcat(colors...))

framerate = 30

record(fig, "Bfield-disk10.gif", collect(1:length(collect(times))); framerate = framerate) do i
    ob.color = colors[i]
    ob.colorrange = (-5, 5)
end

# Record 3D simulation
# tn = Node(0.0)
# sd2 = deepcopy(sd)
# f(t) = begin
#     new_points = [Point3{Float32}(p[1], 0, sol(t)[B_range[i]]) for (i, p) in enumerate(sd[triangle_center(sd), :dual_point])]
#     sd2[triangle_center(sd2), :dual_point] .= new_points
#     sd2
# end

# limits = FRect3D(Vec3f0(-1.0, -1.0, -1.0), Vec3f0(10.0, 10.0, 0.5))
# fig, ax, ob = wireframe(lift(t -> f(t), tn); limits = limits)
# i_range = range(0, 10.0, length = 100)
# framerate = 30
# record(fig, "EM_3D.gif", i_range; framerate = framerate) do i
#     tn[] = i
# end

# t = 10.0
# tn = Node(0.0)
# val = zeros(nv(s))
# norms = zeros(nv(s))
# tris = triangle_vertices(s)
# for i = 1:length(tris[1])
#     for j = 1:3
#         val[tris[j][i]] += sol(t)[i]
#         norms[tris[j][i]] += 1
#     end
# end

# f(t) = begin
#     s2 = deepcopy(s)
#     new_points = [eltype(s2[:point])(p[1], p[2], (val./norms)[i] * 4) for (i, p) in enumerate(s2[:point])]#=exact(p[1], p[2], t; m=0, n=2, A=5, C=1, c=10)=#
#     s2[:point] .= new_points
# end
# limits = FRect3D(Vec3f0(-13.0, -13.0, -13.0), Vec3f0(13.0, 13.0, 13.0))
# fig, ax, ob = wireframe(s2, limits = limits)

# i_range = range(0, 10.0, length = 100)
# framerate = 30
# record(fig, "EM_3D.gif", i_range; framerate = framerate) do i
#     tn[] = i
# end