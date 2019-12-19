# Anushka's dipole mode

using LinearAlgebra, BandedMatrices, StaticArrays, DifferentialEquations
using Plots, ComplexPhasePortrait, Printf

h = 0.5;  N = 7
a = 1.4		# SOR polation

grid = h/2*(1-N:2:N-1) |> SVector{N}
x = reshape(grid,:,1,1)
y = reshape(grid,1,:,1)
z = reshape(grid,1,1,:)

V = r² = abs2.(x) .+ abs2.(y) .+ abs2.(z)
ψ = Complex.(exp.(-r²/2)/√π)

# TODO the right way to determine a broadcast shape
xx = similar(x .+ y .+ z); xx .= x
yy = similar(x .+ y .+ z); yy .= y
zz = similar(x .+ y .+ z); zz .= z


# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

function contract(A,x,dim)
    dims = [dim; setdiff(1:ndims(x), dim)]
    smid = indexin(1:ndims(x), dims)
    B = permutedims(x, dims)
    C = A*reshape(B, size(x, dim), :)
    permutedims(reshape(C, size(B)), smid)
end

# test contraction

for (j,u,uu) = [(1,x,xx), (2,y,yy), (3,z,zz)]
    @assert contract(∂,u,j)[:] ≈ ∂*u[:]
    @assert all(contract(∂,uu,j) .≈ contract(∂,u,j))
end

dψ = contract(∂,ψ,1)
plane = real.(reshape(sum(dψ,dims=2), N, N))

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
