importall POMDPs
using ParticleFilters
using Distributions

struct State
    location::Tuple{Int64,Int64} # (row,column)
    total_battery_used::Float64
    world_map::Array{Bool,2}
end

struct Observation
    obs_battery_used::Float64
    obs_world_map::Array{Tuple{Bool,Float64},2}
end

struct BeliefState
    bel_location::Tuple{Int64,Int64}
    bel_battery_used::Normal{Float64}
    bel_world_map::Array{Float64,2}
end

type UAVpomdp <: POMDPs.POMDP{State,Int64,Observation}
    map_size::Int64
    true_map::Array{Bool,2}
    start_coords::Tuple{Int64,Int64}
    goal_coords::Tuple{Int64,Int64}
    sensor_set::Array{Sensor}
    reward_lambdas::Array{Float64}
end

# Default parameters: map_size is 20 x 20; true_map is all cells true and true_battery_left is 100
UAVpomdp() = UAVpomdp(20, trues(20,20),(1,1),(20,20),[s1::LineSensor,s2::CircularSensor], [1.0,1.0,1.0])

# These seem to be needed by POMCP
# TODO : See if you need discount < 1.0 for POMCP properties
discount(::UAVpomdp) = 1.0
isterminal(p::UAVpomdp,s::State) = (s.coords == p.goal_coords);

# The observation from a state is just the map of th
function generate_o(p::UAVpomdp, s::State, a::Int, sp::State, rng::AbstractRNG)
    
    obs_batt_used = sp.total_battery_used - s.total_battery_used
    obs_map = Array{Tuple{Bool,Float64}}(p.map_size, p.map_size)

    fill!(obs_map,(false,0.0))

    # Should have an enum set for movements and sensors
    if a in MOVEMENTS
        # Just observe current cell with full conf
        obs_map[sp.location[1],sp.location[2]] = ( sp.world_map[sp.location[1],sp.location[2]], 1.0 )
    else
        # action is in SENSORS
        # Need each sensor to implement sense method
        # with arguments (true grid, agent coords) and returns the matrix of obs-conf. Will explain
        # This action is a hindsight thing so does not consume energy
        # TODO : implement sense()
        obs_map = p.sensor_set[a].sense(sp.world_map, sp.location)

    return Observation(obs_batt_used,obs_map)
end

function generate_s(p::UAVpomdp, s::State, a::Int, rng::AbstractRNG)

    # Begin by copying over values
    new_loc = [s.location[1], s.location[2]]
    new_batt = s.total_battery_used
    new_map = s.world_map

    if a in MOVEMENTS
        if a == RIGHT
            # Increase column by one
            new_loc[2] += 1
        else if a == LEFT
            # Decrease column by one
            new_loc[2] -= 1
        else if a == UP
            new_loc[1] -= 1
        else
            new_loc[1] += 1

        if new_loc[1] < 1 || new_loc[1] > p.map_size || new_loc[2] < 1 || new_loc[2] > p.map_size:

            # Just set to old location
            new_loc = [s.location[1], s.location[2]]
    else
        # The only change is to battery
        new_batt += p.sensor_set[a].consumeEnergy(rng)

    return State(new_loc,new_batt,new_map)

# One step reward for various situations
function reward(p::UAVpomdp, s::State, a::Int, sp::State)

    # TODO : Add large positive reward for reaching goal_coords?

    # Component 1 - movement
    # 0 if no movement
    manhattan_distance = abs(s.location[1] - sp.location[1]) + abs(s.location[2]-sp.location[2])
    cost_comp1 = reward_lambdas[1] * manhattan_distance

    # Component 2 - one-step energy usage
    # 0 if no sensing action done
    cost_comp2 = reward_lambdas[2] * (sp.total_battery_used - s.total_battery_used)

    # Component 3 - If in no-fly-zone
    # true means no-fly-zone : additional cost
    cost_comp3 = reward_lambdas[3] * (p.true_map[sp.location(1),sp.location(2)])

    reward = cost_comp1 + cost_comp2 + cost_comp3

    return reward
end

# TODO : Define actions and n_actions and reqd. enums


# Initial distribution with belief state
# Put in some default variables here?
function initial_belief_state(p::UAVpomdp)
    bel_location = (1,1)
    # TODO : Can't have 0.0 std for normal. Shouldn't matter anyway but just
    # make sure it never gets positive reward for negative battery_used
    bel_batt = Distributions.Normal(0.0,eps())
    bel_world_map = 0.5*ones(p.map_size,p.map_size)
    bel_world_map[1,1] = 0.0 # 0% chance of NFZ
end


# generate_s will just use the generate_s of the POMDP
# ParticleFilters - need observation weight function
# Also need a ParticleGenerator?

function obs_weight(p::UAVpomdp, a::Int, sp::State, o::Observation)

    logweight = 0.0
    agreement_weight = 5.0

    # If movement only check that cell of current location is equal to true
    if a in MOVEMENTS
        logweight += log(p.world_map[sp.location[1], sp.location[2]] == o.obs_world_map[sp.location[1], sp.location[2]][1])
    else
        # Loop through cells and update weight based on sensor agreement
        for i = 1 : p.map_size
            for j = 1 : p.map_size

                # TODO : The disagreement will never happen as per our definition - see if that affects anything
                if o.obs_world_map[i,j][1] == sp.world_map[i,j]
                    logweight += o.obs_world_map[i,j][2]*log(agreement_weight)
                else
                    logweight -= o.obs_world_map[i,j][2]*log(agreement_weight)

        # Now weight according to sensor energy agreement
        # This should return the likelihood of energy usage given the sensor's (mean,variance)
        # TODO : Implement this 
        logweight += log(sensor_set[action].energyUsageLikelihood(o.obs_battery_used))

    return exp(logweight)
