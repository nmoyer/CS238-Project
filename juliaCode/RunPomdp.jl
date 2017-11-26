include("GroundTruth.jl")

###############
# Main script #
###############

sim = SimulatorState(40,100)

rng = Base.Random.MersenneTwister(1237)
initial_map = initialize_map(40, 0.3, rng)
state = State([1,1], 0.0, initial_map)
visited_states = [(state.location[1],state.location[2])]
belief_state = initial_belief_state(pomdp)

first_update_simulator(sim, state)

total_reward = 0
action = Int(RIGHT)

n = 1
while true
    update_simulator(sim, state, belief_state)
    
    action = get_action(belief_state, action, "greedy_safe")

    new_state = generate_s(pomdp, state, action, rng)
    push!(visited_states,(new_state.location[1],new_state.location[2]))

    total_reward += reward(pomdp, state, action, new_state)

    obs = generate_o(pomdp, state, action, new_state, rng)
    belief_state = update_belief(pomdp, belief_state, action, obs)

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