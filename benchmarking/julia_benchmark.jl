using Catalyst
using JumpProcesses

rn = @reaction_network begin
    @species Da(t) Dr(t) Dpa(t) Dpr(t) Ma(t) Mr(t) A(t) R(t) C(t)
    @parameters αA αpA αR αpR βA βR δMA δMR δA δR γA γR γC θA θR
    (γA, θA),   Da + A ↔ Dpa
    (γR, θR),   Dr + A ↔ Dpr
    αA,         Da → Da + Ma
    αR,         Dr → Dr + Mr
    αpA,        Dpa → Dpa + Ma
    αpR,        Dpr → Dpr + Mr
    βA,         Ma → Ma + A
    βR,         Mr → Mr + R
    γC,         A + R → C
    δA,         C → R
    δMA,        Ma → ∅
    δMR,        Mr → ∅
    δA,         A → ∅
    δR,         R → ∅
end

p = [50., 500., 0.01, 50., 50., 5., 10., 0.5, 1., 0.2, 1., 1., 2., 50., 100.]
u0 = [1, 1, 0, 0, 0, 0, 0, 0, 0]
tspan = (0., 200.)
prob_discrete = DiscreteProblem(rn, u0, tspan, p)

n_traj = parse(Int, ARGS[1])

prob_jump_cold = JumpProblem(rn, prob_discrete, Direct(), save_positions=(false, false))
start_cold = time()
for i = 1:n_traj
    sol = solve(prob_jump_cold, SSAStepper(), saveat=1.)
end
time_cold = time() - start_cold

prob_jump_warm = JumpProblem(rn, prob_discrete, Direct(), save_positions=(false, false))
start_warm = time()
for i = 1:n_traj
    sol = solve(prob_jump_warm, SSAStepper(), saveat=1.)
end
time_warm = time() - start_warm

println(time_cold, ",", time_warm)