using MCTS, BasicPOMCP#, POMCPOW
using Base.Profile
include("GroundTruth.jl")

function run_iteration_alg(sim, sensors, lambdas, rng, initial_map, alg, grid_size, suppress_sim)

    if alg == "mdp"

        const solver = MCTSSolver(n_iterations=1000, depth=20, rng=rng)
        const mdp = UAVBeliefMDP(grid_size, initial_map, START_LOC, [grid_size, grid_size], sensors, lambdas)
        policy = solve(solver, mdp)
        initial_belief_map = 0.5*ones(grid_size,grid_size)
        initial_belief_map[START_LOC[1],START_LOC[2]] = 0.0
        state = MDPState(START_LOC, [START_BATTERY,eps()], initial_belief_map)

    elseif alg == "pomcp"

        const solver = POMCPSolver(tree_queries=TREE_QUERIES, c=C, max_depth=MAX_DEPTH, rng=rng)
        const pomdp = UAVpomdp(grid_size, initial_map, START_LOC, [grid_size, grid_size], sensors, lambdas)
        policy = solve(solver, pomdp)
        belief_state = initial_belief_state(pomdp)
        state = State(START_LOC, START_BATTERY, initial_map)

    else 

        const pomdp = UAVpomdp(grid_size, initial_map, START_LOC, [grid_size, grid_size], sensors, lambdas)
        belief_state = initial_belief_state(pomdp)
        state = State(START_LOC, START_BATTERY, initial_map)

    end

    if !suppress_sim
        first_update_simulator(sim, initial_map)
    end

    total_reward = 0
    iteration = 0
    solved = false
    while iteration < 1000
        if !suppress_sim
            if alg == "mdp"
                update_simulator(sim, state)
            else
                update_simulator(sim, state, belief_state)
            end
        end

        if alg == "mdp"
            a = action(policy, state)
        elseif alg == "pomcp"
            a = action(policy, belief_state)
        else
            a = greedy_information_action(pomdp, belief_state)
        end

        if alg == "mdp" 
            new_state, reward = next_state_reward_true(mdp, state, a, rng)
        else
            new_state = generate_s(pomdp, state, a, rng)
            reward = reward_no_heuristic(pomdp, state, a, new_state)
            obs = generate_o(pomdp, state, a, new_state, rng)
            belief_state = update_belief(pomdp, belief_state, a, obs)
        end
        
        total_reward += reward
        state = new_state

        if alg == "mdp"
            if isterminal(mdp, state)
                if !suppress_sim
                    update_simulator(sim, state)
                end
                solved=true
                break
            end
        else
            if isterminal(pomdp, state)
                if !suppress_sim
                    update_simulator(sim, state, belief_state)
                end
                solved=true
                break
            end
        end

        iteration += 1
    end

    return total_reward,solved
end 


function run_trials(sim, sensors, lambdas, num_trials, suppress_sim, start_seed, alg, grid_size, percent_obstruct)

    oracle_cost = -100000000000.0
    initial_map = 0
    free_path = 1
    cp_start_seed = start_seed

    while free_path > 0
        cp_start_seed += 1
        rng = Base.Random.MersenneTwister(cp_start_seed)
        initial_map = initialize_map(grid_size, percent_obstruct, rng)
        free_path, oracle_cost = cost_of_oracle(initial_map, START_LOC, [grid_size, grid_size], lambdas)
    end

    rewards = []
    n_no_sols = 0

    for seed in cp_start_seed:cp_start_seed + (num_trials-1)

        rng = Base.Random.MersenneTwister(seed)
        reward,solved = run_iteration_alg(sim, sensors, lambdas, rng, initial_map, alg, grid_size, suppress_sim)
        push!(rewards, reward)
        if solved == false
            n_no_sols +=1
        end
    end

    avg_rewards = sum(rewards) ./ num_trials
    std_dev_rewards = (sum([(reward-avg_rewards)^2 for reward in rewards])/num_trials)^(0.5)

    return avg_rewards, std_dev_rewards, oracle_cost, n_no_sols
end

function run_trials_with_alg(sensors, lambdas, NUM_TRIALS, SUPPRESS_SIM, START_SEED, alg)

    print("Trying with "*alg*"\n")
    for grid_size = 10:5:20
        for percent_obstruct = 0.1:0.1:0.3

            if !SUPPRESS_SIM
                sim = SimulatorState(grid_size,0)
            else
                sim = 0
            end

            print("***Grid size="*string(grid_size)*",percent_obstructed="*string(percent_obstruct)*"***\n")
            avg_cost, std_dev_cost, oracle_cost, n_no_sols = run_trials(sim, sensors, lambdas, NUM_TRIALS, SUPPRESS_SIM, START_SEED, alg, grid_size, percent_obstruct)
            print("Avg reward - "*string(avg_cost)"\n")
            fname = alg*"-"*string(grid_size)*"-"*string(percent_obstruct)*".txt"
            open(fname,"w") do f
                write(f,"$avg_cost, $std_dev_cost, $oracle_cost, $n_no_sols")
            end
        end
    end

end

####################
# INPUT PARAMETERS #
####################

#const GRID_SIZE = 15
#const PERCENT_OBSTRUCT = 0.2

const START_LOC = [1,1]
const START_BATTERY = 0.0

const MOVEMENT_LAMBDA = 1.0
const HEURISTIC_LAMBDA = 1.0
const SENSOR_LAMBDA = 1.0
const NFZ_LAMBDA = 20
const SUCCESS_LAMBDA = 1000.0

const SUPPRESS_SIM = true

const sensors = [LineSensor([0,1]),LineSensor([1,0]),LineSensor([0,-1]),LineSensor([-1,0]),CircularSensor()]
const lambdas = [MOVEMENT_LAMBDA, HEURISTIC_LAMBDA, SENSOR_LAMBDA, NFZ_LAMBDA, SUCCESS_LAMBDA]

const TREE_QUERIES = 1000
const C = 1.0
const MAX_DEPTH = 40

const NUM_TRIALS = 20
const START_SEED = 300

for algorithm in ["mdp","pomcp","greedy"]
    run_trials_with_alg(sensors, lambdas, NUM_TRIALS, SUPPRESS_SIM, START_SEED, algorithm)
end

# if !suppress_sim
#     freeze_simulator(sim)
# end