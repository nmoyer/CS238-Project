using MCTS, BasicPOMCP#, POMCPOW
include("GroundTruth.jl")

function run_iteration(sim, state, belief_state, )

###############
# Main script #
###############

GRID_SIZE = 15

# Default parameters: map_size is 20 x 20; true_map is all cells true and true_battery_left is 100
# TODO : Cost of NFZ should be HIGH to encourage sensor usage
sensors = [LineSensor([0,1]),LineSensor([1,0]),LineSensor([0,-1]),LineSensor([-1,0]),CircularSensor()]
lambdas = [1.0,1.0,15.0]
pomdp = UAVpomdp(GRID_SIZE, falses(GRID_SIZE,GRID_SIZE), [1,1], [GRID_SIZE,GRID_SIZE], sensors, lambdas)

solver = POMCPSolver(tree_queries=1000, max_depth=30)
policy = solve(solver, pomdp);

sim = SimulatorState(GRID_SIZE,0)
rng = Base.Random.MersenneTwister(1245)

initial_map = initialize_map(GRID_SIZE, 0.3, rng)
state = State([1,1], 0.0, initial_map)
pomdp.true_map = initial_map
belief_state = initial_belief_state(pomdp)

first_update_simulator(sim, state)

total_reward = 0
n = 1

while true
    update_simulator(sim, state, belief_state)

    a = action(policy, belief_state)    

    new_state = generate_s(pomdp, state, a, rng)
    total_reward += reward(pomdp, state, a, new_state)
    obs = generate_o(pomdp, state, a, new_state, rng)
    belief_state = update_belief(pomdp, belief_state, a, obs)

    state = new_state

    if isterminal(pomdp, state)
        update_simulator(sim, state, belief_state)
        break
    end

    n += 1
end 

print("Final score:")
print(total_reward)
print("\n")

freeze_simulator(sim)