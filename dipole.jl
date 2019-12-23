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

dR = hz*hxy^2
∫(u) = dR*sum(u)

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
∂² = (1/hxy^2).*op(Nxy, [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
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
    @assert contract(∂²,u,j)[:] ≈ ∂²*u[:]
    @assert all(contract(∂²,uu,j) .≈ contract(∂²,u,j))
end
@assert contract(∂²z,z,3)[:] ≈ ∂²z*z[:]
@assert all(contract(∂²z,zz,3) .≈ contract(∂²z,z,3))

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
# -∇²*ψ/2-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ = μ*ψ

ψ = copy(TFψ)
ψ₀ = similar(ψ)
residual = []
for _ = 1:100
    ψ₀ .= ψ
    for R = keys(ψ)
        i,j,k = Tuple(R)
        ψ[R] = 0
        T = (∂²[i,:]⋅ψ[:,j,k]+∂²[j,:]⋅ψ[i,:,k]+∂²z[k,:]⋅ψ[i,j,:]) / 2
        ψr = (μ*ψ₀[R]+T) / (-∂²[1,1,1]-∂²z[1,1,1]/2+V[R]+g*(abs2.(ψ₀[R])))
        ψ[R] = ψ₀[R] + a*(ψr-ψ₀[R])
     end
     Lψ = -(contract(∂²,ψ,1)+contract(∂²,ψ,2)+contract(∂²z,ψ,3))/2 +
         V.*ψ + g*(abs2.(ψ)).*ψ
     E = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
     push!(residual, norm(Lψ-E*ψ)/norm(ψ))
end

# scatter(residual, mc=:black, ms=2, yscale=:log10, leg=:none)

# # propagate
# 
# P = ODEProblem((ψ,p,t)->-1im*(-∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))), ψ, (0.0,10.0))
# @time S = solve(P; saveat=0.5)
# 
# zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
# zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
