using MCTS, BasicPOMCP#, POMCPOW
using Base.Profile
include("GroundTruth.jl")

# @ NateM : The solver here will be the ARDESPOT object that you create in the outer level
#           You probably do not need to touch the other arguments
function run_iteration(sim, solver, sensors, lambdas, seed, suppress_sim)

    rng = Base.Random.MersenneTwister(seed)
    solver = POMCPSolver(tree_queries=TREE_QUERIES,c=C, max_depth=MAX_DEPTH, rng=rng)

    const initial_map = initialize_map(GRID_SIZE, PERCENT_OBSTRUCT, rng)
    const pomdp = UAVpomdp(GRID_SIZE, initial_map, START_LOC, END_LOC, sensors, lambdas)
    
    const naive_cost = cost_of_naive(pomdp)
    const oracle_cost = cost_of_oracle(pomdp)
    print(string(oracle_cost)*"\n")

    # if oracle_cost < (10000-28)
    #     return 0
    # end

    policy = solve(solver, pomdp)
    belief_state = initial_belief_state(pomdp)
    state = State(START_LOC, START_BATTERY, initial_map)

    if !suppress_sim
        first_update_simulator(sim, state)
    end

    total_reward = 0
    iteration = 0
    while iteration < 1000000
        if !suppress_sim
            update_simulator(sim, state, belief_state)
        end

        #a = action(policy, belief_state) 
        a = greedy_information_action(pomdp, belief_state)
        print("TAKING ACTION"*string(a)*"\n")

        new_state = generate_s(pomdp, state, a, rng)
        total_reward += reward_no_heuristic(pomdp, state, a, new_state)
        obs = generate_o(pomdp, state, a, new_state, rng)
        belief_state = update_belief(pomdp, belief_state, a, obs)
        state = new_state

        if isterminal(pomdp, state)
            if !suppress_sim
                update_simulator(sim, state, belief_state)
            end 

            break
        end

        iteration += 1
    end

    print(total_reward - naive_cost)
    return total_reward - naive_cost
end 

function run_trials(sim, solver, sensors, lambdas, num_trials, suppress_sim, start_seed)

    all_rewards = 0.0
    for seed in start_seed:start_seed + (num_trials-1)
        all_rewards += @time run_iteration(sim, solver, sensors, lambdas, seed, suppress_sim)
    end
    average_reward = all_rewards ./ num_trials

    # @ Nate M : Depending on what works for you, you might want to print to a file here
    #            So it is easier to accumulate results later
    print("Average reward: "*string(average_reward)*"\n")

    if !suppress_sim
        freeze_simulator(sim)
    end
end

####################
# INPUT PARAMETERS #
####################

const GRID_SIZE = 30
const PERCENT_OBSTRUCT = 0.4

const START_LOC = [1,1]
const END_LOC = [GRID_SIZE, GRID_SIZE]
const START_BATTERY = 0.0

const MOVEMENT_LAMBDA = 1.0
const HEURISTIC_LAMBDA = 2.0
const SENSOR_LAMBDA = 2.0
const NFZ_LAMBDA = 30.0
const SUCCESS_LAMBDA = 10000.0

const SUPPRESS_SIM = false

if !SUPPRESS_SIM
    sim = SimulatorState(GRID_SIZE,0)
else
    sim = 0
end

const sensors = [LineSensor([0,1]),LineSensor([1,0]),LineSensor([0,-1]),LineSensor([-1,0]),CircularSensor()]
const lambdas = [MOVEMENT_LAMBDA, HEURISTIC_LAMBDA, SENSOR_LAMBDA, NFZ_LAMBDA, SUCCESS_LAMBDA]

const TREE_QUERIES = 1000
const C = 1.0
const MAX_DEPTH = 40

solver = POMCPSolver(tree_queries=TREE_QUERIES,c=C, max_depth=MAX_DEPTH)

const NUM_TRIALS = 100
const START_SEED = 100

#Base.Profile.init

run_trials(sim, solver, sensors, lambdas, NUM_TRIALS, SUPPRESS_SIM, START_SEED)

#Base.Profile.print()