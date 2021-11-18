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
    neg₁::Hom(Form1(X), Form1(X)) # In/Out edge flow boundary condition
    E::Hom(munit(), Form1(X))     # electric field
    B::Hom(munit(), Form2(X))     # magnetic field
    c⁻¹::Hom(Form2(X), Form2(X))   # μ₀ ⋅ ϵ₀ inverse (scalar)
end


# Define Electromagnetic physics
@present EM2D <: EM2DQuantities begin
    B ⋅ ∂ₜ(Form2(X)) == E ⋅ neg₁ ⋅ d₁(X)
    E ⋅ ∂ₜ(Form1(X)) == B ⋅ c⁻¹ ⋅ ⋆₂⁻¹(X) ⋅ dual_d₂(x) ⋅ ⋆₁(X)
end

diag = eq_to_diagrams(EM2D)
to_graphviz(diag; edge_len = "1.3")

dwd = diag2dwd(diag)
to_graphviz(dwd, orientation = LeftToRight)

s = EmbeddedDeltaSet2D("../meshes/wire.obj")
sd = dual(s);

# Define non-default operators (multiplication by constants)
c⁻¹ = 1 / (1.2566e-6 * 8.8542e-12)
funcs = sym2func(sd)

funcs[:c⁻¹] = Dict(:operator => :c⁻¹ * I(ne(sd)), :type => MatrixFunc())
funcs[:neg₁] = Dict(:operator => -1 * I(ne(sd)), :type => MatrixFunc())

# Define initial conditions
x = [p[1] for p in s[:point]];

eField(x) = begin
    amp = sin((3) * π * (x[1]))
    amp * Point{3,Float32}(1, 0, 0)
end

E = ♭(sd, DualVectorField(eField.(sd[triangle_center(sd), :dual_point])))

B = TriForm(zeros(ntriangles(s)))

u0 = vcat(E.data, B.data)

# # Compress wiring diagram over contiguous matrix multiplications
cont_dwd = deepcopy(dwd)
Examples.contract_matrices!(cont_dwd, funcs)

# # Generate simulation function
func, _ = gen_sim(cont_dwd, funcs, sd; autodiff = false);

# # Solve problem
prob = ODEProblem(func, u0, (0, 5))
sol = solve(prob, Tsit5());