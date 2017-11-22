importall POMDPs
using ParticleFilters
using Distributions
include("Sensors.jl")

#################################
# Define state and action space #
#################################

# TODO : Work with StaticArrays for location?
struct State
    location::Array{Int64,1} # (row,column)
    total_battery_used::Float64
    world_map::BitArray
end

struct Observation
    obs_location::Array{Int64,1}
    obs_battery_used::Float64
    obs_world_map::Array{Tuple{Bool,Float64},2}
end

struct BeliefState
    bel_location::Array{Int64,1}
    bel_battery_used::Tuple{Float64,Float64}
    bel_world_map::Array{Float64,2}
end

#Define actions by ints
NUM_SENSORS = 2
NUM_MOVEMENTS = 4

SENSORS = 1:NUM_SENSORS
MOVEMENTS = NUM_SENSORS+1:NUM_SENSORS+NUM_MOVEMENTS

@enum MOVEMENT_STRING RIGHT=NUM_SENSORS+1 LEFT=NUM_SENSORS+2 UP=NUM_SENSORS+3 DOWN=NUM_SENSORS+4

# The observation from a state is just the map of th
function generate_o(p::UAVpomdp, s::State, a::Int, sp::State, rng::AbstractRNG)
    
    obs_loc = sp.location
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
    end

    return Observation(obs_loc,obs_batt_used,obs_map)
end

function generate_s(p::Type{UAVpomdp}, s::State, a::Int, rng::MersenneTwister)

    # Begin by copying over values
    new_loc = [s.location[1], s.location[2]]
    new_batt = s.total_battery_used
    new_map = s.world_map

    if a in MOVEMENTS
        if a == Int(RIGHT)
            # Increase column by one
            new_loc[2] += 1
        elseif a == Int(LEFT)
            # Decrease column by one
            new_loc[2] -= 1
        elseif a == Int(UP)
            new_loc[1] -= 1
        else
            new_loc[1] += 1
        end

        if new_loc[1] < 1 || new_loc[1] > p.map_size || new_loc[2] < 1 || new_loc[2] > p.map_size
            # Just set to old location
            new_loc = [s.location[1], s.location[2]]
        end
    else
        # The only change is to battery
        new_batt += p.sensor_set[a].consumeEnergy(rng)
    end

    return State(new_loc,new_batt,new_map)
end

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

    reward = -(cost_comp1 + cost_comp2 + cost_comp3)

    return reward
end

###############################################
# Define POMDP struct and initialize instance #
###############################################

type UAVpomdp <: POMDPs.POMDP{State,Int64,Observation}
    map_size::Int64
    true_map::BitArray{2}
    start_coords::Array{Int64,1}
    goal_coords::Array{Int64,1}
    sensor_set::Array{Sensor,1}
    reward_lambdas::Array{Float64}
end

# These seem to be needed by POMCP
# TODO : See if you need discount < 1.0 for POMCP properties
discount(::UAVpomdp) = 1.0
isterminal(p::UAVpomdp,s::State) = (s.coords == p.goal_coords);

# Default parameters: map_size is 20 x 20; true_map is all cells true and true_battery_left is 100
# TODO : Cost of NFZ should be HIGH to encourage sensor usage
UAVpomdp() = UAVpomdp(20, falses(20,20),(1,1),(20,20),[s1::LineSensor,s2::CircularSensor], [1.0,1.0,1.0])

# Initial distribution with belief state
# Put in some default variables here?
function initial_belief_state(p::UAVpomdp)
    bel_location = [1,1]
    # TODO : Can't have 0.0 std for normal. Shouldn't matter anyway but just
    # make sure it never gets positive reward for negative battery_used
    bel_batt = (0.0,eps())
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
                end
            end
        end
        # Now weight according to sensor energy agreement
        # This should return the likelihood of energy usage given the sensor's (mean,variance)
        # TODO : Implement this 
        logweight += log(sensor_set[action].energyUsageLikelihood(o.obs_battery_used))
    end

    return exp(logweight)
end


# The explicit belief updated to be used by the outer Loop
function update_belief(p::UAVpomdp,b::BeliefState,a::Int,o::Observation)

    # New belief location is just the observed location
    # Initialize battery dist with old normal
    # Initialize obs_map with old map
    new_bel_loc = o.obs_location
    new_batt_used = b.bel_battery_used
    new_bel_map = b.bel_world_map

    # If action is movement, only the 
    # bel_map for the new cell changes
    if a in MOVEMENTS
        # Implicit conversion from bool to double
        new_bel_map[o.obs_location[1],o.obs_location[2]] = o.obs_world_map[o.obs_location[1],o.obs_location[2]]
    else
        # Action is a sensor usage
        # Update battery gaussians
        # TODO - make mean and std of energy directly accessible as tuple of floats
        sensor_mean_std = p.sensor_set[a].energyMeanSTD()
        new_batt_used = (b.bel_battery_used[1] + sensor_mean_std[1] , sqrt(b.bel_battery_used[2]^2 + sensor_mean_std[2]^2))


        # Now update grid
        for i = 1 : p.map_size
            for j = 1 : p.map_size

                if o.obs_world_map[i,j][2] > 0.0
                    confval = 0.0
                    weightcoeff = 1.0 #TODO - This weights the new observation vs current
                    
                    # Do a weighted sum of old and new estimates
                    if o.obs_world_map[i,j][1] == true
                        confval = (b.bel_world_map[i,j] + 1.0*weightcoeff)/(1.0 + o.obs_world_map[i,j][2]*weightcoeff)
                    else
                        confval = (b.bel_world_map[i,j])/(1.0 + o.obs_world_map[i,j][2]*weightcoeff)
                    
                    new_bel_map[i,j] = confval

    return BeliefState(new_bel_loc, new_batt_used, new_bel_map)



# A sampler for the belief state
function rand(rng::AbstractRNG, b::BeliefState)

    state_loc = [b.bel_location[1], b.bel_location[2]]
    state_bat = rand(rng,b.bel_battery_used)

    # Now initialize grid
    # Sample a random number for each cell and assign cell state based on result
    state_map = BitArray{2}(p.map_size,p.map_size)
    for i = 1 : p.map_size
        for j = 1 : p.map_size

            randval = rand(rng)
            if randval < b.bel_world_map[i,j]
                state_map[i,j] = true
            else
                state_map[i,j] = false

    return State(state_loc,state_bat,state_map)


