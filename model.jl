##The model is based on the implementation by Paul Schrimpf (https://faculty.arts.ubc.ca/pschrimpf)
#For now, this is an implementation on simulation data.

using Distributions, DataFrames

"""
         brentrymodel(data::AbstractDataFrame,
                      n::Symbol,
                      s::Symbol,
                      x::Array{Symbol,1},
                      w::Array{Symbol,1};
                      Fϵ)

Create loglikelihood for Bresnehan & Reiss style entry model

Inputs:
- `data` DataFrame
- `n` name of number of firm variable in data
- `s` name of market size variable in data
- `x` array of names of variable profit shifters
- `w` array of names of fixed cost shifters
- `Fϵ` cdf of ϵ, optional, defaults to standard normal cdf

The same variables may be included in both `x` and `w`.
"""
function brentrymodel(data::AbstractDataFrame,
                      n::Symbol,
                      s::Symbol,
                      x::Array{Symbol,1},
                      w::Array{Symbol,1};
                      Fϵ = x->cdf(Normal(),x))
  # skip observations with missings
  vars = unique([n, s, x..., w...])
  inc = completecases(data[vars])

  N = disallowmissing(data[n][inc])
  S = disallowmissing(data[s][inc])
  X = disallowmissing(convert(Matrix, data[x][inc,:]))
  W = disallowmissing(convert(Matrix, data[w][inc,:]))
  Nmax = maximum(N)
  function packparam(α,β,γ,δ)
    θ = [α;β;γ;δ]
  end
  function unpackparam(θ)
    α = θ[1:Nmax]
    β = θ[(Nmax+1):(Nmax+size(X,2))]
    γ = θ[(Nmax+size(X,2)+1):(Nmax+size(X,2)+Nmax)]
    δ = θ[(Nmax+size(X,2)+Nmax+1):end]
    (α,β,γ,δ)
  end

  # While maximizing the likelihood some parameters might result in
  # the likelihood being 0 (or very close to 0) taking log would
  # create problems. Use logfinite from PharmacyEntry.jl instead
  logf = logfinite(exp(-100.0) ) # could adjust the exp(-100.0)

  function loglike(θ)
    (α,β,γ,δ) = unpackparam(θ)
    error("You must write the body of this function")
    # P = array of likelihoods for each observation
    # return(mean(logf.(P)))
  end

  return(loglike=loglike, unpack=unpackparam, pack=packparam)
end

# Simulating data
using DataFrames, Statistics, StatsBase
import CSV
df = CSV.read("cleandata.csv")

# Important to scale variables to avoid numerical problems in both
# simulation & estimation
df[:pop10k] = df[Symbol("Population, 2016")]./10000
df[:logpop10k] = log.(df[:pop10k])
df[:income10k] = df[Symbol("Average total income in 2015 among recipients (\$)")]./10000
df[:density1k] = df[Symbol("Population density per square kilometre")]./1000
df[:logdensity] = log.(df[:density1k])
df[:logarea] = log.(df[Symbol("Land area in square kilometres")])
df[:mediumage] = df[Symbol("15 to 64 years")]./100
# parameters for simulation
n_obs_sim = 500 # you might want to adjust this. You want it to be
                # large enough that your estimates are close to the
                # true values, but small enough that it doesn't take
                # too long to estimate

# the maximum number of pharmacies in the simulated data will be
# length(α) + 1
α = [1.0, -1.]
γ = [1.0,  1.]
# you may have to adjust the parameters to get a reasonable distribution of
# number of pharmacies across markets
svar = :pop10k
β = [1., 1.]
xvars = [:income10k,
         :mediumage]
δ = [1., 1.]
wvars = [:logdensity,
         :logarea]
simdf = df[sample(1:nrow(df), n_obs_sim),:]

simdf[:nsim] = brentrysim(simdf, svar, xvars, wvars, α,β,γ,δ)
println("Distribution of number of firms")

for i in 0:length(α)
  println("$(mean(simdf[:nsim].==i))")
end

using Optim, ForwardDiff, LinearAlgebra, PrettyTables
try
  using EntrySolution
  # this contains my code for the likelihood and
  # it's intentionally not included in the assignment
catch
end

(loglike, unpack, pack) = brentrymodel(simdf, :nsim, svar, xvars, wvars)
θ0 = pack(α,β,γ,δ)
loglike(θ0)

# initial values --- note that you may run into optimization problems
# with poor initial values. This is especially likely if
# s*cumsum(α)[c] - cumsum(γ)[c] is not decreasing with c. You can
# ensure this by making α < 0 and γ>0
βi = zeros(size(β))
δi = zeros(size(δ))
αi = zeros(size(α))
γi = ones(size(γ))
θi = pack(αi, βi, γi, δi);
loglike(θi)

res = optimize((x)->(-loglike(x)), θi, method=BFGS(),
               autodiff=:forward, show_trace=true)


               # if you have problems, maybe look at one parameter at a time, e.g.
               # res = optimize((x)->(-loglike(pack(x, β, γ, δ))), αi, method=BFGS(), autodiff=:forward, show_trace=true)
               θhat = res.minimizer
               (αhat, βhat, γhat, δhat) = unpack(θhat)

               # calculate standard errors
               H = ForwardDiff.hessian(loglike,θhat)
               Varθ = -inv(H)./nrow(simdf);
               (seα, seβ, seγ, seδ) = unpack(sqrt.(diag(Varθ)))

               # Print a nice(ish) table
               header= ["Parameter", "Truth", "Estimate", "(SE)"];
               param = [["α[$i]" for i in eachindex(α)];
                        ["β[$i]" for i in eachindex(β)];
                        ["γ[$i]" for i in eachindex(γ)];
                        ["δ[$i]" for i in eachindex(δ)]];
               # highlight estimates that reject H0 : estimate = true at 99% level
               h1 = Highlighter(
                 f = (tbl, i, j)->( (j==3 || j==4) &&
                                  abs((tbl[i,2]-tbl[i,3])/tbl[i,4]).>quantile(Normal(),
                                                                              0.995)),
                 crayon = crayon"red bold"
               );
