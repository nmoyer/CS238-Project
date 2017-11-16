importall POMDPs

#include("State.jl") # The struct for maintaining the state, as we discussed
#include("Sensor.jl") # Should have a base class and a bunch of derived classes
#include("Observation.jl") # Struct has a binary map and battery usage amount

type UAVpomdp <: POMDPs.POMDP{State,Int64,Observation}
    map_size::Int64
    true_map::Array{Bool,2}
    true_battery_left::Float64
    start_coords::Tuple{Int64,Int64}
    goal_coords::Tuple{Int64,Int64}
end

# Default parameters: map_size is 20 x 20; true_map is all cells true and true_battery_left is 100
UAVpomdp() = UAVpomdp(10, trues(10,10), 100.0,(1,1),(20,20))

# These seem to be needed by POMCP
discount(::UAVpomdp) = 1.0
isterminal(p::UAVpomdp,s::State) = (s.coords == p.goal_coords);