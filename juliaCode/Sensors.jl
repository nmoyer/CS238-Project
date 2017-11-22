
##################
# Define sensors #
##################

abstract type Sensor
end

LINE_SENSOR_ENERGY_USE = 2
# TODO Distributions.Normal works with std not variance
LINE_SENSOR_ENERGY_VAR = 0.1
LINE_SENSOR_LENGTH = 5
LINE_SENSOR_WIDTH = 1
LINE_SENSOR_MAX_CONF = 0.7

type LineSensor <: Sensor
    sense::Function
    consumeEnergy::Function
    energyUsageLikelihood::Function

    function LineSensor()
        instance = new()
 
        instance.sense = function (world_map::BitArray, loc::Array{Int64,1})
            confidence_stepsize = LINE_SENSOR_MAX_CONF/LINE_SENSOR_LENGTH
            confidences = LINE_SENSOR_MAX_CONF:0:-confidence_stepsize

            obs_map = Array{Tuple{Bool,Float64}}(p.map_size, p.map_size)
            fill!(obs_map,(false,0.0))

            # TODO : Check boundary conditions in array
            for i = 1:LINE_SENSOR_WIDTH
                for j = 1:LINE_SENSOR_LENGTH
                    row = loc[0] + i
                    col = loc[1] + j
                    observed_bool = true #placeholder
                    observations[row][col] = (observed_bool, confidences[j])
                end
            end
        end

        instance.consumeEnergy = function (rng::AbstractRNG)
            return LINE_SENSOR_ENERGY_USE
        end
 
        # TODO : Needs to return a value not a distribution
        instance.energyUsageLikelihood = function (obs_battery_used::Float64)
            return Distributions.Normal(LINE_SENSOR_ENERGY_USE,
                                        LINE_SENSOR_ENERGY_VAR)
        end

        return instance
    end
end

CIRCULAR_SENSOR_ENERGY_USE = 4
CIRCULAR_SENSOR_ENERGY_VAR = 0.2
CIRCULAR_SENSOR_RADIUS = 3
CIRCULAR_SENSOR_MAX_CONF = 0.9

CIRCULAR_OFFSETS = Dict(1 => [[0, 1], [1, 0], [0, -1], [-1, 0]],
                        2 => [[0, 2], [1, 1], [0, 0], [-1, 1], 
                              [2, 0], [1, -1], [0, 0], [0, -2], 
                              [-1, -1], [-2, 0]]
                        3 => [[0, 3], [1, 2], [0, 1], [-1, 2], 
                              [2, 1], [1, 0], [0, 1], [0, -1], 
                              [-1, 0], [-2, 1], [3, 0], [2, -1], 
                              [1, 0], [1, -2], [0, -1], [-1, 0], 
                              [0, -3], [-1, -2], [-2, -1], [-3, 0]]
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
            loc[0] = center[0] + offsets[0]
            loc[1] = center[1] + offsets[1]
            if loc[0] > map_len || loc[0] <= 0 || loc[1] > map_len || loc[1] <= 0
                continue
            push!(locations,loc)
        end
    end
end

struct CircularSensor <: Sensor
    sense::Function
    consumeEnergy::Function
    energyUsageLikelihood::Function

    function CircularSensor()
        instance = new()
 
        instance.sense = function (world_map::BitArray, loc::Array{Int64,1})
            confidence_stepsize = LINE_SENSOR_MAX_CONF/LINE_SENSOR_LENGTH
            confidences = LINE_SENSOR_MAX_CONF:0:-confidence_stepsize
          
            obs_map = Array{Tuple{Bool,Float64}}(p.map_size, p.map_size)
            fill!(obs_map,(false,0.0))

            for (row, col, d) in generate_circle(loc, CIRCULAR_SENSOR_RADIUS)
                observed_bool = true #placeholder
                observations[row][col] = (observed_bool, confidences[d])
            end
        end
 
        instance.consumeEnergy = function (rng::AbstractRNG)
            return CIRCULAR_SENSOR_ENERGY_USE
        end
 
        instance.energyUsageLikelihood = function (obs_battery_used::Float64)
            return Distributions.Normal(CIRCULAR_SENSOR_ENERGY_USE,
                                        CIRCULAR_SENSOR_ENERGY_VAR)
        end

        return instance
    end
end