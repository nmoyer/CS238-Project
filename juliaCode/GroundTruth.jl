using PyCall
include("UAVpomdp.jl")

@pyimport Tkinter as tk

##################################
# Simulator struct and functions #
##################################

struct SimulatorState
	tk_root::PyObject
	canvas::PyObject
	canvas_true_grid::Array{Array{Int64,1},1}
    canvas_belief_grid::Array{Array{Int64,1},1}
    canvas_battery::Array{Int64,1}

    function SimulatorState(grid_world_size::Int64, total_battery::Int64)
        tk_root = tk.Tk()

        grid_tile_size = 15

        adj_canvas_size = grid_world_size*grid_tile_size
        battery_canvas_space = Int(floor(adj_canvas_size/20))*4
        final_width = 2*adj_canvas_size+battery_canvas_space

        canvas = tk.Canvas(tk_root, width=final_width, height=adj_canvas_size)
        canvas[:grid](row=0,column=0)

        ground_truth_start = adj_canvas_size + Int(floor(battery_canvas_space/3))
        ground_truth_end = (2*adj_canvas_size) + Int(floor(battery_canvas_space/3))-1
        canvas_true_grid = [[canvas[:create_rectangle](col, row, col+grid_tile_size, 
                                                  row+grid_tile_size, fill="white") 
                        for col in ground_truth_start:grid_tile_size:ground_truth_end] 
                        for row in 1:grid_tile_size:adj_canvas_size]

        canvas_belief_grid = [[canvas[:create_rectangle](col, row, col+grid_tile_size, 
                                                  row+grid_tile_size, fill="gray50") 
                        for col in 1:grid_tile_size:adj_canvas_size] 
                        for row in 1:grid_tile_size:adj_canvas_size]

        # battery_canvas_start = 2*adj_canvas_size + 2*Int(floor(battery_canvas_space/3)) 
        # battery_canvas_end = 2*adj_canvas_size + 3*Int(floor(battery_canvas_space/3))
        # battery_tile_size = Int(floor(adj_canvas_size/total_battery))

        # canvas_battery = [canvas[:create_rectangle](battery_canvas_start,row,
        #                                             battery_canvas_end,
        #                                             row+battery_tile_size, fill="green") 
        #                   for row in 0:battery_tile_size:adj_canvas_size]

        return new(tk_root, canvas, canvas_true_grid, canvas_belief_grid)#, canvas_battery)
    end
end

function first_update_simulator(sim::SimulatorState, initial_state::State)

    canvas = sim.canvas
    canvas_true_grid = sim.canvas_true_grid    
    world_map = initial_state.world_map
    map_size = size(canvas_true_grid,1)

    for row = 1:map_size
        for col = 1:map_size
            if world_map[row,col]
                tile_belief = 1.0
            else
                tile_belief = 0.0
            end
            canvas[:itemconfig](canvas_true_grid[row][col], fill=get_color(tile_belief))
        end
    end

end

function update_simulator(sim::SimulatorState, state::State, belief_state::BeliefState)

    canvas_true_grid = sim.canvas_true_grid
    canvas_belief_grid = sim.canvas_belief_grid
    canvas_battery = sim.canvas_battery
    canvas = sim.canvas
    tk_root = sim.tk_root
    bel_world_map = belief_state.bel_world_map
    location = state.location
    total_battery_used = state.total_battery_used

    map_size = size(canvas_belief_grid,1)

    for row = 1:map_size
        for col = 1:map_size
            tile_belief = bel_world_map[row,col]
            canvas[:itemconfig](canvas_belief_grid[row][col], fill=get_color(tile_belief))
        end
    end

    aircraft_row, aircraft_col = location
    canvas[:itemconfig](canvas_true_grid[aircraft_row][aircraft_col], fill="blue")
    canvas[:itemconfig](canvas_belief_grid[aircraft_row][aircraft_col], fill="blue")

    # for level = 0:total_battery_used
    #     canvas[:itemconfig](canvas_battery[Int(level)+1], fill="white")
    # end

    tk_root[:update]()
    #sleep(.5)
end

function freeze_simulator(sim::SimulatorState)
    
    tk_root = sim.tk_root
    tk_root[:mainloop]()

end

########################
# Grid world functions #
########################

function make_cluster(world_map::BitArray{2}, rng::MersenneTwister)
    const cluster_height = 1
    const cluster_width = 1

    row_start = Base.Random.rand(rng, -cluster_height:size(world_map,1))
    col_start = Base.Random.rand(rng, -cluster_width:size(world_map,2))

    for row = row_start:row_start+cluster_height
        for col = col_start:col_start+cluster_width
            if row > size(world_map,1) || row <= 0 || col > size(world_map,2) || col <= 0
                continue
            end
            world_map[row,col] = true
        end
    end
end

function perc_obstruct(world_map::BitArray{2})
    cum_sum = 0.0
    for i in indices(world_map,1)
        for j in indices(world_map,2)
            cum_sum += world_map[i,j]
        end
    end
    return cum_sum/(size(world_map,1)*size(world_map,2))
end

function initialize_map(map_size::Int64, desired_perc::Float64, rng::MersenneTwister)
    world_map = falses(map_size,map_size)
    while perc_obstruct(world_map) < desired_perc
        make_cluster(world_map, rng)
    end
    return world_map
end

function get_color(tile_belief::Float64)
    if tile_belief == 1.0
        return "black"
    elseif tile_belief == 0.0
        return "white"
    end

    gray_num = Int(floor(100 - tile_belief*100))
    return "gray" * string(gray_num)
end

function cost_of_naive(p::UAVpomdp)
    total_cost = 0
    for i = 1:p.map_size
        row1 = i
        col1 = i
        row2 = i
        col2 = i+1

        total_cost -= 2*p.reward_lambdas[1]
        
        if p.true_map[row1, col1] && i != 1
            total_cost -= p.reward_lambdas[4]
        end 
        if i != p.map_size && p.true_map[row2, col2]
            total_cost -= p.reward_lambdas[4]
        end

        total_cost
    end

    total_cost += p.reward_lambdas[5]

    return total_cost
end

function get_action(belief_state::BeliefState, prev_action::Int64, policy_type::String)
    if policy_type == "greedy_safe"

        if prev_action > 5
            return 5
        end

        map_size = size(belief_state.bel_world_map,1)
        loc = belief_state.bel_location
        moves = [(0,-1), (1,0), (0,1), (-1,0)]

        min_bel = 1.0
        min_move = (0,0)
        for move in moves
            row = loc[1]+move[1]
            col = loc[2]+move[2]

            if (row <= 0) || (col <= 0) || (row > map_size) || (col > map_size)
                continue
            end

            move_bel = belief_state.bel_world_map[row,col]

            if (row, col) in visited_states
                move_bel += 0.2
            end

            if move == (1,0) || move == (0,1)
                move_bel -= 0.1
            end

            if move_bel < min_bel
                min_bel = move_bel
                min_move = move
            end
        end

        if min_move == (0,-1)
            return Int(LEFT)
        elseif min_move == (0,1)
            return Int(RIGHT)
        elseif min_move == (-1,0)
            return Int(UP)
        elseif min_move == (1,0)
            return Int(DOWN)
        else
            print("No max move")
            return Int(RIGHT)
        end

    else

        if prev_action == 6
            return 9
        else
            return 6
        end
    end
end

# ###############
# # Main script #
# ###############

# sim = SimulatorState(40,100)

# rng = Base.Random.MersenneTwister(1237)
# initial_map = initialize_map(40, 0.3, rng)
# state = State([1,1], 0.0, initial_map)
# visited_states = [(state.location[1],state.location[2])]
# belief_state = initial_belief_state(pomdp)

# first_update_simulator(sim, state)

# total_reward = 0
# action = Int(RIGHT)

# n = 1
# while true
# 	update_simulator(sim, state, belief_state)
    
#     action = get_action(belief_state, action, "greedy_safe")

#     new_state = generate_s(pomdp, state, action, rng)
#     push!(visited_states,(new_state.location[1],new_state.location[2]))

#     total_reward += reward(pomdp, state, action, new_state)

#     obs = generate_o(pomdp, state, action, new_state, rng)
#     belief_state = update_belief(pomdp, belief_state, action, obs)

#     state = new_state

#     if isterminal(pomdp, state)
#         update_simulator(sim, state, belief_state)
#         break
#     end

#     n += 1
# end 

    # if rand(rng,Float64) < 0.5
    #     a = 5 
    # else
    #     if rand(rng,Float64) < 0.5
    #         a = Int(DOWN)
    #     else
    #         a = Int(RIGHT)
    #     end
    # end

# print("Final score:")
# print(total_reward)
# print("\n")

# freeze_simulator(sim)