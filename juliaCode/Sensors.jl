using Distributions

##################
# Define sensors #
##################

function sigmoid(z::Float64)
    return 1.0 ./ (1.0 .+ exp(-z))
end

function expected_prob_NFZ(prob_right::Float64, prob_NFZ::Float64) 
    prob_obs_NFZ = prob_NFZ*prob_right + (1-prob_NFZ)*(1-prob_right)
    prob_not_obs_NFZ = prob_right*(1-prob_NFZ) + (1-prob_right)*prob_NFZ

    posterior_obs_NFZ = (prob_right*prob_NFZ)./prob_obs_NFZ
    posterior_not_obs_NFZ = ((1-prob_right)*(1-prob_NFZ))./prob_not_obs_NFZ

    return prob_obs_NFZ*posterior_obs_NFZ + prob_not_obs_NFZ*posterior_not_obs_NFZ
end

function delta_expected_confidence(prob_right::Float64, prob_NFZ::Float64) 
    prob_obs_NFZ = prob_NFZ*prob_right + (1-prob_NFZ)*(1-prob_right)
    prob_not_obs_NFZ = prob_right*(1-prob_NFZ) + (1-prob_right)*prob_NFZ

    if prob_NFZ < 0.01 || prob_NFZ > 0.99
        return 0
    end 

    #print("\nProb\n"*string(prob_obs_NFZ)*","*string(prob_not_obs_NFZ)*"\n")

    change_if_obs_nfz = abs(((prob_right*prob_NFZ)./prob_obs_NFZ) - prob_NFZ)
    change_if_not_obs_nfz = abs((((1-prob_right)*(1-prob_NFZ))./prob_not_obs_NFZ) - prob_NFZ)

    #print("Change\n"*string(change_if_obs_nfz)*","*string(change_if_not_obs_nfz)*"\n\n")

    return prob_obs_NFZ*change_if_obs_nfz + prob_not_obs_NFZ*change_if_not_obs_nfz
end

abstract type Sensor
end

const LINE_SENSOR_ENERGY_USE = 1
const LINE_SENSOR_ENERGY_SD = 0.1
const LINE_SENSOR_LENGTH = 10
const LINE_SENSOR_WIDTH = 1
const LINE_SENSOR_MAX_CONF = 0.7
const LINE_SENSOR_MIN_CONF = 0.5

function generate_line(line_start::Array{Int64,1}, len::Int64, direction::Array{Int64,1}, 
                       map_len::Int64)

    locations = Array{Int64, 1}[]
    
    for i = 1:len
        loc = [line_start[1],line_start[2],i]

        if direction[1] != 0 
            loc[1] = line_start[1] + direction[1]*i
        else 
            loc[2] = line_start[2] + direction[2]*i   
        end 

        if loc[1] > map_len || loc[1] <= 0 || loc[2] > map_len || loc[2] <= 0
            continue
        end
        push!(locations,loc)
    end

    return locations
end

type LineSensor <: Sensor
    sense::Function
    changeConfidence::Function
    updateBelMapMDP::Function
    consumeEnergy::Function
    energyUsageLikelihood::Function
    energySpec::Tuple{Float64, Float64}

    function LineSensor(direction::Array{Int64,1})
        sense = function (world_map::BitArray, loc::Array{Int64,1}, rng::MersenneTwister)
            map_size = size(world_map, 1)
            confidence_stepsize = (LINE_SENSOR_MAX_CONF-LINE_SENSOR_MIN_CONF)/LINE_SENSOR_LENGTH
            confidences = LINE_SENSOR_MAX_CONF:-confidence_stepsize:LINE_SENSOR_MIN_CONF

            obs_map = Array{Tuple{Bool,Float64}}(map_size, map_size)
            fill!(obs_map,(false,0.0))

            for loc in generate_line(loc, LINE_SENSOR_LENGTH, direction, map_size)
                row, col, d = loc
                prob_right = confidences[d]
                if rand(rng,Float64) < prob_right
                    observed_bool = world_map[row, col]
                else
                    observed_bool = !world_map[row, col]
                end
                new_val = (observed_bool, confidences[d])
                setindex!(obs_map, new_val, row, col)
            end

            return obs_map
        end

        changeConfidence = function (belief_map::Array{Float64,2}, loc::Array{Int64,1})
            map_size = size(belief_map, 1)
            confidence_stepsize = (LINE_SENSOR_MAX_CONF-LINE_SENSOR_MIN_CONF)/LINE_SENSOR_LENGTH
            confidences = LINE_SENSOR_MAX_CONF:-confidence_stepsize:LINE_SENSOR_MIN_CONF

            delta_confidence = 0
            for loc in generate_line(loc, LINE_SENSOR_LENGTH, direction, map_size)
                row, col, d = loc
                prob_right = confidences[d]       
                delta_confidence += delta_expected_confidence(prob_right, belief_map[row,col]) 
            end

            return delta_confidence
        end 

        updateBelMapMDP = function (belief_map::Array{Float64,2},loc::Array{Int64,1},rng::MersenneTwister)
            map_size = size(belief_map, 1)
            confidence_stepsize = (LINE_SENSOR_MAX_CONF-LINE_SENSOR_MIN_CONF)/LINE_SENSOR_LENGTH
            confidences = LINE_SENSOR_MAX_CONF:-confidence_stepsize:LINE_SENSOR_MIN_CONF

            new_belief_map = deepcopy(belief_map)

            for loc in generate_line(loc, LINE_SENSOR_LENGTH, direction, map_size)
                row, col, d = loc
                prob_right = confidences[d]
                prob_NFZ = belief_map[row,col]

                prob_obs_NFZ = prob_NFZ*prob_right + (1.0 - prob_NFZ)*(1.0 - prob_right)
                #print(string(prob_obs_NFZ)*"\n")

                if rand(rng,Float64) < prob_obs_NFZ
                    top = prob_right*prob_NFZ
                    bottom = top + (1.0 - prob_right)*(1.0 - prob_NFZ)
                    new_belief_map[row,col] = top ./ bottom
                else
                    top = (1.0 - prob_right)*prob_NFZ
                    bottom = top + prob_right*(1.0 - prob_NFZ)
                    new_belief_map[row,col] = top ./ bottom 
                end
            end

            return new_belief_map
        end

        consumeEnergy = function (rng::AbstractRNG)
            distribution = Normal(LINE_SENSOR_ENERGY_USE,
                                  LINE_SENSOR_ENERGY_SD)
            return rand(distribution)#,rng)
        end
 
        energyUsageLikelihood = function (obs_battery_used::Float64)
            distribution = Normal(LINE_SENSOR_ENERGY_USE,
                                  LINE_SENSOR_ENERGY_SD)
            return pdf(distribution,obs_battery_used)
        end

        energySpec = (LINE_SENSOR_ENERGY_USE, LINE_SENSOR_ENERGY_SD)
        return new(sense, changeConfidence, updateBelMapMDP, consumeEnergy, energyUsageLikelihood, energySpec)
    end
end

const CIRCULAR_SENSOR_ENERGY_USE = 4
const CIRCULAR_SENSOR_ENERGY_SD = 0.2
const CIRCULAR_SENSOR_RADIUS = 4
const CIRCULAR_SENSOR_MAX_CONF = 0.9
const CIRCULAR_SENSOR_MIN_CONF = 0.7

const CIRCULAR_OFFSETS = Dict(1 => [[0, 1], [1, 0], [0, -1], [-1, 0]],
                        2 => [[0, 2], [1, 1], [0, 0], [-1, 1], 
                              [2, 0], [1, -1], [0, 0], [0, -2], 
                              [-1, -1], [-2, 0]], 
                        3 => [[0, 3], [1, 2], [0, 1], [-1, 2], 
                              [2, 1], [1, 0], [0, 1], [0, -1], 
                              [-1, 0], [-2, 1], [3, 0], [2, -1], 
                              [1, 0], [1, -2], [0, -1], [-1, 0], 
                              [0, -3], [-1, -2], [-2, -1], [-3, 0]],
                        4 => [[0, 4], [1, 3], [0, 2], [-1, 3], 
                              [2, 2], [1, 1], [0, 2], [0, 0], 
                              [-1, 1], [-2, 2], [3, 1], [2, 0], 
                              [1, 1], [1, -1], [0, 0], [-1, 1], 
                              [0, -2], [-1, -1], [-2, 0], [-3, 1], 
                              [4, 0], [3, -1], [2, 0], [2, -2], 
                              [1, -1], [0, 0], [1, -3], [0, -2], 
                              [-1, -1], [-2, 0], [0, -4], [-1, -3], 
                              [-2, -2], [-3, -1], [-4, 0]])

function generate_circle(center::Array{Int64,1}, radius::Int64, map_len::Int64)
    locations = Array{Int64, 1}[]

    for r in 1:radius
        offsets = CIRCULAR_OFFSETS[r]
        for offset in offsets
            loc = [0,0,r]
            loc[1] = center[1] + offset[1]
            loc[2] = center[2] + offset[2]
            if loc[1] > map_len || loc[1] <= 0 || loc[2] > map_len || loc[2] <= 0
                continue
            end
            push!(locations,loc)
        end
    end

    return locations
end

type CircularSensor <: Sensor
    sense::Function
    changeConfidence::Function
    updateBelMapMDP::Function
    consumeEnergy::Function
    energyUsageLikelihood::Function
    energySpec::Tuple{Float64, Float64}

    function CircularSensor()
        sense = function (world_map::BitArray, loc::Array{Int64,1}, rng::MersenneTwister)
            map_size = size(world_map, 1)
            confidence_stepsize = (CIRCULAR_SENSOR_MAX_CONF-CIRCULAR_SENSOR_MIN_CONF)/CIRCULAR_SENSOR_RADIUS
            confidences = CIRCULAR_SENSOR_MAX_CONF:-confidence_stepsize:CIRCULAR_SENSOR_MIN_CONF

            obs_map = Array{Tuple{Bool,Float64}}(map_size, map_size)
            fill!(obs_map,(false,0.5))

            for loc in generate_circle(loc, CIRCULAR_SENSOR_RADIUS, map_size)
                row, col, d = loc
                prob_right = confidences[d]
                if rand(rng,Float64) < prob_right
                    observed_bool = world_map[row, col]
                else
                    observed_bool = !world_map[row, col]
                end
                new_val = (observed_bool, confidences[d])
                setindex!(obs_map, new_val, row, col)
            end

            return obs_map
        end
 
        changeConfidence = function (belief_map::Array{Float64,2}, loc::Array{Int64,1})
            map_size = size(belief_map, 1)
            confidence_stepsize = (CIRCULAR_SENSOR_MAX_CONF-CIRCULAR_SENSOR_MIN_CONF)/CIRCULAR_SENSOR_RADIUS
            confidences = CIRCULAR_SENSOR_MAX_CONF:-confidence_stepsize:CIRCULAR_SENSOR_MIN_CONF

            delta_confidence = 0
            for loc in generate_circle(loc, CIRCULAR_SENSOR_RADIUS, map_size)
                row, col, d = loc
                prob_right = confidences[d]       
                delta_confidence += delta_expected_confidence(prob_right, belief_map[row,col]) 
            end

            return delta_confidence
        end 

        updateBelMapMDP = function (belief_map::Array{Float64,2},loc::Array{Int64,1},rng::MersenneTwister)
            map_size = size(belief_map, 1)
            confidence_stepsize = (CIRCULAR_SENSOR_MAX_CONF-CIRCULAR_SENSOR_MIN_CONF)/CIRCULAR_SENSOR_RADIUS
            confidences = CIRCULAR_SENSOR_MAX_CONF:-confidence_stepsize:CIRCULAR_SENSOR_MIN_CONF

            new_belief_map = deepcopy(belief_map)

            for loc in generate_circle(loc, CIRCULAR_SENSOR_RADIUS, map_size)
                row, col, d = loc
                prob_right = confidences[d]
                prob_NFZ = belief_map[row,col]

                prob_obs_NFZ = prob_NFZ*prob_right + (1.0 - prob_NFZ)*(1.0 - prob_right)

                if rand(rng,Float64) < prob_obs_NFZ
                    top = prob_right*prob_NFZ
                    bottom = top + (1.0 - prob_right)*(1.0 - prob_NFZ)
                    new_belief_map[row,col] = top ./ bottom
                else
                    top = (1.0 - prob_right)*prob_NFZ
                    bottom = top + prob_right*(1.0 - prob_NFZ)
                    new_belief_map[row,col] = top ./ bottom 
                end
            end
 
            return new_belief_map
        end

        consumeEnergy = function (rng::AbstractRNG)
            distribution = Normal(CIRCULAR_SENSOR_ENERGY_USE,
                                  CIRCULAR_SENSOR_ENERGY_SD)
            return rand(distribution)#,rng)
        end
 
        energyUsageLikelihood = function (obs_battery_used::Float64)
            distribution = Normal(CIRCULAR_SENSOR_ENERGY_USE,
                                  CIRCULAR_SENSOR_ENERGY_SD)
            return pdf(distribution,obs_battery_used)
        end

        energySpec = (CIRCULAR_SENSOR_ENERGY_USE, CIRCULAR_SENSOR_ENERGY_SD)
        return new(sense, changeConfidence, updateBelMapMDP, consumeEnergy, energyUsageLikelihood, energySpec)
    end
end