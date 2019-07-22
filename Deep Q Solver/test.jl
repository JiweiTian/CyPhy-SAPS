push!(LOAD_PATH, ".")
    include("pomdp.jl")

    using Sz, Reduction, Grid_class, Wind, PowerModels, JLD, DeepQLearning
    using LightGraphs, Random, Parameters, RLInterface, Flux, POMDPs, StatsBase
    using POMDPModelTools, POMDPModels, POMDPSimulators, Convex

    PowerModels.silence()

shortest_path, W, obj, contingencies = Wind.wind_simulation("case24_ieee_rts")

contingencies

num_buses = length(obj.rnc["bus"])

sz = length(obj.E[:,1])
for i in 1:sz
    obj.E = vcat(obj.E, [obj.E[i,2] obj.E[i,1]])
end

ACTION_SET = pomdp_class.create_set(shortest_path[1], num_buses)
initial_state = Random.rand(1:num_buses)
initial_actions = ACTION_SET[initial_state]
initial_prev_action = initial_actions[Random.rand(1:length(initial_actions))]
c1 = shortest_path[2][1]
c2 = shortest_path[2][end]

rewards_arr = pomdp_class.reward_calculator(obj.E, W, c1, c2, obj, contingencies)

pomdp_problem = pomdp_class.PowerGridEnv(ACTION_SET, initial_state, initial_prev_action, initial_actions, false, false,
                     c1, c2, rewards_arr, [], .2)

model = Chain(Dense(1, length(ACTION_SET)), Dense(length(ACTION_SET), length(ACTION_SET)))

solver = pomdp_class.DeepQLearningSolver(qnetwork = model, max_steps=1000,
                             learning_rate=0.005,log_freq=500, double_q=false,
                             dueling=false)

env = POMDPEnvironment(pomdp_problem, rng=solver.rng)

policy = pomdp_class.NNPolicy(pomdp_problem, model,
                                pomdp_class.actions(pomdp_problem),
                                length(pomdp_class.obs_dimensions(pomdp_problem)))

solved = pomdp_class.solve(solver, env, model, policy)

simul = HistoryRecorder(max_steps = 49)

up = pomdp_class.HistoryUpdater(pomdp_problem)
r = simulate(simul, pomdp_problem, solved[1], up)

x = 44

contingencies[1] in r.state_hist[1:x]
contingencies[2] in r.state_hist[1:x]

contingencies[3] in r.state_hist[1:x]
contingencies[4] in r.state_hist[1:x]

# for i in r.state_hist[1:45]
#     print(i, ", ")
# end

# using GraphPlot
# nodelabel = [i for i in 1:num_buses]
# gplot(shortest_path[1], nodelabel = nodelabel)

push!(LOAD_PATH, ".")
include("pomdp.jl")

using Sz, Reduction, Grid_class, Wind, PowerModels, JLD, DeepQLearning
using LightGraphs, Random, Parameters, RLInterface, Flux, POMDPs, StatsBase
using POMDPModelTools, POMDPModels, POMDPSimulators, Convex

casename = "case89pegase"
case = string("cases/", casename, ".m")
rnc = PowerModels.parse_file(case)
rnc2 = PowerModels.parse_file(case)
obj = Grid_class.Grid_class_const(rnc, rnc2, casename)
# obj.f = obj.f[:,1]

obj, y = Grid_class.N_1_analysis(obj)
Grid_class.N_2_analysis(obj, "fast")


# findmax(y[:,3])
#
# # y[14,:]
#
#
# obj.E[53,:]
#
# findall(x->x["t_bus"]==26, obj.rnc["branch"])
#
# obj.rnc["branch"]["27"]
#
# obj.rc_original["branch"]["27"]
