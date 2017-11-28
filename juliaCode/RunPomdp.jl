using MCTS, BasicPOMCP#, POMCPOW
include("GroundTruth.jl")

function run_iteration(sim, solver, sensors, lambdas, seed, suppress_sim)

    rng = Base.Random.MersenneTwister(seed)

    initial_map = initialize_map(GRID_SIZE, PERCENT_OBSTRUCT, rng)
    pomdp = UAVpomdp(GRID_SIZE, initial_map, START_LOC, END_LOC, sensors, lambdas)
    policy = solve(solver, pomdp)
    belief_state = initial_belief_state(pomdp)
    state = State(START_LOC, START_BATTERY, initial_map)

    if !suppress_sim
        first_update_simulator(sim, state)
    end

    total_reward = 0
    while true
        if !suppress_sim
            update_simulator(sim, state, belief_state)
        end

        a = action(policy, belief_state) 
        new_state = generate_s(pomdp, state, a, rng)
        total_reward += reward(pomdp, state, a, new_state)
        obs = generate_o(pomdp, state, a, new_state, rng)
        belief_state = update_belief(pomdp, belief_state, a, obs)
        state = new_state

        if isterminal(pomdp, state)
            if !suppress_sim
                update_simulator(sim, state, belief_state)
            end 

            break
        end
    end

    return total_reward
end 

function run_trials(sim, solver, sensors, lambdas, num_trials, suppress_sim)

    all_rewards = 0.0
    for seed in 1:num_trials
        all_rewards += run_iteration(sim, solver, sensors, lambdas, seed, suppress_sim)
    end
    average_reward = all_rewards ./ num_trials

    print("Average reward: "*string(average_reward)*"\n")

    if !suppress_sim
        freeze_simulator(sim)
    end
end

####################
# INPUT PARAMETERS #
####################

GRID_SIZE = 15
PERCENT_OBSTRUCT = 0.3

START_LOC = [1,1]
END_LOC = [GRID_SIZE, GRID_SIZE]
START_BATTERY = 0.0

MOVEMENT_LAMBDA = 1.0
SENSOR_LAMBDA = 1.0
NFZ_LAMBDA = 15.0

SUPPRESS_SIM = true

if !SUPPRESS_SIM
    sim = SimulatorState(GRID_SIZE,0)
else
    sim = 0
end

sensors = [LineSensor([0,1]),LineSensor([1,0]),LineSensor([0,-1]),LineSensor([-1,0]),CircularSensor()]
lambdas = [MOVEMENT_LAMBDA, SENSOR_LAMBDA, NFZ_LAMBDA]

TREE_QUERIES = 1000

solver = POMCPSolver(tree_queries=TREE_QUERIES)

NUM_TRIALS = 3

run_trials(sim, solver, sensors, lambdas, NUM_TRIALS, SUPPRESS_SIM)
