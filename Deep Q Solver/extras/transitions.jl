function transition_probability(obj)
  av_values = [.85, .62, .55, .2]

  exploitability_dict = Dict()

  exploitability_dict["G"] = Dict("AC"=>.44, "PR"=>.5, "UI"=>.62, "S"=> "changed", "AV"=>av_values[Random.rand(1:4)])
  exploitability_dict["W"] = Dict("AC"=>.44, "PR"=>.68, "UI"=>.62, "S"=> "changed", "AV"=>av_values[Random.rand(1:4)])
  exploitability_dict["L"] = Dict("AC"=>.77, "PR"=>.27, "UI"=>.85, "S"=> "unchanged", "AV"=>av_values[Random.rand(1:4)])
  exploitability_dict["N"] = Dict("AC"=>.77, "PR"=>.62, "UI"=>.85, "S"=> "unchanged", "AV"=>av_values[Random.rand(1:4)])

  gen_list = [obj.rnc["gen"]["$i"]["gen_bus"] for (i,gen) in obj.rnc["gen"]]
  wind_list = [40]
  load_list = [obj.rnc["load"]["$i"]["load_bus"] for (i,load) in obj.rnc["load"]]
  c_i_a_values  = [.56; .22; 0]

  E_arr = Float64[]
  I_arr = Float64[]
  transition_arr = Float64[]

  for i in 1:length(obj.E[:,1])
     if obj.E[i,2] in gen_list
         mult_dict = exploitability_dict["G"]
     elseif obj.E[i,2] in wind_list
         mult_dict = exploitability_dict["W"]
     elseif obj.E[i,2] in load_list
         mult_dict = exploitability_dict["L"]
     else
         mult_dict = exploitability_dict["N"]
     end

     max_val = 0
     E = 0
     I = 0

     for i in av_values
         for j in c_i_a_values
              for k in c_i_a_values
                   for l in c_i_a_values

                     E = 8.22 * i * mult_dict["AC"] * mult_dict["PR"] * mult_dict["UI"]
                     I_base = (1 - (1 - j) * (1 - k) * (1 - l))

                     if mult_dict["S"] == "changed"
                         I = 7.52 * (I_base - 0.029) - (3.25 * ((I_base - 0.02) ^ 15))
                     else
                         I = 6.42 * I_base
                     end
                 end
             end
         end
     end

     if mult_dict["S"] == "changed"
         B = min(1.08 * (E + I), 10)
     else
         B = min((E + I), 10)
     end
     push!(transition_arr, E/B)
  end
  return transition_arr
end
