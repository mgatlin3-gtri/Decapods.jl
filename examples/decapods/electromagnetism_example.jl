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
    B ⋅ ∂ₜ(Form2(X)) == neg₁ ⋅ ⋆₁(X) ⋅ d₁(X) ⋅ E
    E ⋅ ∂ₜ(Form1(X)) == ⋆₂⁻¹ ⋅ B ⋅ c⁻¹
end

s = EmbeddedDeltaSet2D("../meshes/wire.obj")
sd = dual(s);

# Define non-default operators (multiplication by constants)
c⁻¹ = 1 / (1.2566 * 8.8542)
funcs = sym2func(sd)

funcs[:c⁻¹] = Dict(:operator => :c⁻¹ * I(ne(sd)), :type => MatrixFunc())
funcs[:neg₁] = Dict(:operator => -1 * I(ne(sd)), :type => MatrixFunc())

# Define initial conditions
