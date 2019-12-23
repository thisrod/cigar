# Anushka's dipole mode

using LinearAlgebra, BandedMatrices, StaticArrays, DifferentialEquations
using Plots, ComplexPhasePortrait, Printf

N = 1e5		# number of atoms
g = 0.06194	# repulsion constant
hxy = 0.5;  Nxy = 21
hz = 5; Nz = 17
a = 1.4		# SOR polation

grid = hxy/2*(1-Nxy:2:Nxy-1) |> SVector{Nxy}
x = reshape(grid,:,1,1)
y = reshape(grid,1,:,1)
grid = hz/2*(1-Nz:2:Nz-1) |> SVector{Nz}
z = reshape(grid,1,1,:)

xyplane(u) = heatmap(y[:], x[:], u[:,:,(Nz+1)÷2], xlabel="y", ylabel="x")
xzplane(u) = heatmap(z[:], x[:], u[:,(Nxy+1)÷2,:], xlabel="z", ylabel="x")
∫(u) = hz*hxy^2*sum(u)

V = (abs2.(x) .+ abs2.(y) .+ abs2.(z)/69.91)/2

# TODO the right way to determine a broadcast shape
xx = similar(x .+ y .+ z); xx .= x
yy = similar(x .+ y .+ z); yy .= y
zz = similar(x .+ y .+ z); zz .= z

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(N, stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/hxy).*op(Nxy, [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/hxy^2).*op(Nxy, [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
∂z = (1/hz).*op(Nz, [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂²z = (1/hz^2).*op(Nz, [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

function contract(A,x,dim)
    dims = [dim; setdiff(1:ndims(x), dim)]
    smid = indexin(1:ndims(x), dims)
    B = permutedims(x, dims)
    C = A*reshape(B, size(x, dim), :)
    permutedims(reshape(C, size(B)), smid)
end

# test contraction

for (j,u,uu) = [(1,x,xx), (2,y,yy)]
    @assert contract(∂,u,j)[:] ≈ ∂*u[:]
    @assert all(contract(∂,uu,j) .≈ contract(∂,u,j))
end
@assert contract(∂z,z,3)[:] ≈ ∂z*z[:]
@assert all(contract(∂z,zz,3) .≈ contract(∂z,z,3))

# Thomas-Fermi oprm to find healing length and grid size
μ₀ = 0.0
μ₁ = 20.0
TFψ = similar(V)
for _ = 1:1000
    global μ₀, μ₁
    μ = μ₀ + (μ₁-μ₀)/2
    TFψ .= sqrt.(max.(0, μ .- V)/g)
    NTF = ∫(abs2.(TFψ))
    if abs(NTF-N) < 0.1
        break
    elseif NTF > N
        μ₁ = μ
    else
        μ₀ = μ
    end
end
μ = μ₀ + (μ₁-μ₀)/2

# solve by SOR: 
# -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ = μ*ψ

# ψ₀ = similar(ψ)
# nnc = zero(ψ)	# left in for future use
# potential = []
# residual = []
# @time for _ = 1:Int(1000)
#     ψ₀ .= ψ
#     for k = keys(ψ)
#         i,j = Tuple(k)
#         ψ[k] = 0
#         T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
#         L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
#         ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
#             (-2*∂²[1,1]+V[k]+C*(abs2.(ψ₀[k])+2nnc[k]))
#         ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
#         impose_vortex!(ψ)
#      end
#      Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*(abs2.(ψ₀)+2nnc).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
#      # take residual of unconstrained components
#      impose_vortex!(Lψ)
#      E = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
#      push!(potential, E)
#      push!(residual, norm(Lψ-E*ψ)/norm(ψ))
# end
# 
# # propagate
# 
# P = ODEProblem((ψ,p,t)->-1im*(-∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))), ψ, (0.0,10.0))
# @time S = solve(P; saveat=0.5)
# 
# zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
# zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
