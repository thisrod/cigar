# Anushka's dipole mode

using LinearAlgebra, BandedMatrices, StaticArrays, DifferentialEquations
using Plots, ComplexPhasePortrait

N = 1e5		# number of atoms
g = 0.06194	# repulsion constant
b = 0.5		# trap to cloud offset

hx = 0.5;  Nx = 35
hy = 0.5;  Ny = 21
hz = 5; Nz = 17
a = 1.4		# SOR polation

grid(h,N) = h/2*(1-N:2:N-1) |> SVector{N}
x = reshape(grid(hx,Nx),:,1,1)
y = reshape(grid(hy,Ny),1,:,1)
z = reshape(grid(hz,Nz),1,1,:)

zplot(h,v,u) = plot(v, h,
    portrait(reverse(u,dims=1)).*abs2.(u)/maximum(abs2.(u)),
    ratio=:equal)
zplot(h,v,u::Matrix{<:Real}) = zplot(h,v,Complex.(u))

xyplane(u) = (
    zplot(x[:], y[:], u[:,:,(Nz+1)÷2]);
    xlabel!("y"); ylabel!("x");
)
xzplane(u) = (
    zplot(x[:], z[:], u[:,(Ny+1)÷2,:]);
    xlabel!("z"); ylabel!("x");
)

H = hx*hy*hz
∫(u) = H*sum(u)

# Trap for dynamics
V = (abs2.(x) .+ abs2.(y) .+ abs2.(z)/69.91)/2

# Notional trap for initial order parameter
V₀ = (abs2.(x.-b) .+ abs2.(y) .+ abs2.(z)/69.91)/2

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
∂²x = (1/hx^2).*op(Nx, [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
∂²y = (1/hy^2).*op(Ny, [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
∂²z = (1/hz^2).*op(Nz, [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

function contract(A,x,dim)
    dims = [dim; setdiff(1:ndims(x), dim)]
    smid = indexin(1:ndims(x), dims)
    B = permutedims(x, dims)
    C = A*reshape(B, size(x, dim), :)
    permutedims(reshape(C, size(B)), smid)
end

# test contraction
@assert contract(∂²x,x,1)[:] ≈ ∂²x*x[:]
@assert all(contract(∂²x,xx,1) .≈ contract(∂²x,x,1))
@assert contract(∂²y,y,2)[:] ≈ ∂²y*y[:]
@assert all(contract(∂²y,yy,2) .≈ contract(∂²y,y,2))
@assert contract(∂²z,z,3)[:] ≈ ∂²z*z[:]
@assert all(contract(∂²z,zz,3) .≈ contract(∂²z,z,3))

# Thomas-Fermi oprm to find healing length and grid size
μ₀ = 0.0
μ₁ = 20.0
TFψ = similar(V₀)
for _ = 1:1000
    global μ₀, μ₁
    μ = μ₀ + (μ₁-μ₀)/2
    TFψ .= sqrt.(max.(0, μ .- V₀)/g)
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
# -∇²*ψ/2-ψ*∂²+V₀.*ψ+C*abs2.(ψ).*ψ = μ*ψ

ψ = copy(TFψ)
ψ₀ = similar(ψ)
residual = []
diagel = -(∂²x[1,1,1]+∂²y[1,1,1]+∂²z[1,1,1])/2
for _ = 1:100
    ψ₀ .= ψ
    for R = keys(ψ)
        i,j,k = Tuple(R)
        ψ[R] = 0
        T = (∂²x[i,:]⋅ψ[:,j,k]+∂²y[j,:]⋅ψ[i,:,k]+∂²z[k,:]⋅ψ[i,j,:]) / 2
        ψr = (μ*ψ₀[R]+T) / (diagel+V₀[R]+g*(abs2.(ψ₀[R])))
        ψ[R] = ψ₀[R] + a*(ψr-ψ₀[R])
     end
     Lψ = -(contract(∂²x,ψ,1)+contract(∂²y,ψ,2)+contract(∂²z,ψ,3))/2 +
         V₀.*ψ + g*(abs2.(ψ)).*ψ
     E = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
     push!(residual, norm(Lψ-E*ψ)/norm(ψ))
end

# scatter(residual, mc=:black, ms=2, yscale=:log10, leg=:none)

# propagate

T = 10.0
P = ODEProblem((ψ,_,_)->
    -1im*(-(contract(∂²x,ψ,1)+contract(∂²y,ψ,2)+contract(∂²z,ψ,3))/2 + 
        (V.-μ).*ψ + g*abs2.(ψ).*ψ),
    Complex.(ψ), (0.0,T))
S1 = solve(P)
Q = ODEProblem((ψ,_,_)->
    -1im*(-(contract(∂²x,ψ,1)+contract(∂²y,ψ,2)+contract(∂²z,ψ,3))/2 + 
        (V.-μ).*ψ + g*abs2.(ψ).*ψ),
    Complex.(TFψ), (0.0,T))
S2 = solve(Q)

tt = 0.0:0.1:T
ev(A,φ) = [∫(A.*abs2.(φ(t))) / ∫(abs2.(φ(t)))  for t in tt]

# plot(tt,ev(x,S), lc=:black, leg=:none)
# plot!(tt,hypot.(ev(y,S),ev(z,S)), lc=:red, leg=:none)
# plot(tt,ev(x.^2,S).-ev(x,S).^2, lc=:black, leg=:none)
# plot(tt,ev(y.^2,S).-ev(y,S).^2, lc=:black, leg=:none)
# plot(tt,ev(z.^2,S).-ev(z,S).^2, lc=:black, leg=:none)
