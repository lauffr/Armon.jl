
function write_data_to_file(params::ArmonParameters, data::ArmonDataOrDual,
        col_range, row_range, file; direct_indexing=false, for_3D=true)
    @indexing_vars(params)

    vars_to_write = saved_variables(data isa ArmonDualData ? host(data) : data)

    p = params.output_precision
    format = Printf.Format(join(repeat(["%#$(p+7).$(p)e"], length(vars_to_write)), ", ") * "\n")

    # TODO: center the positions of the cells

    for j in col_range
        for i in row_range
            if direct_indexing
                idx = i + j - 1
            else
                idx = @i(i, j)
            end

            Printf.format(file, format, getindex.(vars_to_write, idx)...)
        end

        for_3D && println(file)  # Separate rows to use pm3d plotting with gnuplot
    end
end


function build_file_path(params::ArmonParameters, file_name::String)
    (; output_dir, use_MPI, is_root, cart_coords) = params

    file_path = joinpath(output_dir, file_name)

    if is_root && !isdir(output_dir)
        mkpath(output_dir)
    end

    if use_MPI
        (cx, cy) = cart_coords
        params.use_MPI && (file_path *= "_$(cx)x$(cy)")
    end

    return file_path
end


function write_sub_domain_file(params::ArmonParameters, data::ArmonDataOrDual, file_name::String; no_msg=false)
    (; silent, nx, ny, is_root, nghost) = params

    output_file_path = build_file_path(params, file_name)

    open(output_file_path, "w") do file
        col_range = 1:ny
        row_range = 1:nx
        if params.write_ghosts
            col_range = inflate(col_range, nghost)
            row_range = inflate(row_range, nghost)
        end

        write_data_to_file(params, data, col_range, row_range, file)
    end

    if !no_msg && is_root && silent < 2
        println("\nWrote to files $(output_file_path)_*x*")
    end
end


function write_slices_files(params::ArmonParameters, data::ArmonDataOrDual, file_name::String; no_msg=false)
    (; output_dir, silent, nx, ny, use_MPI, is_root, cart_comm, global_grid, proc_dims, cart_coords) = params

    if is_root && !isdir(output_dir)
        mkpath(output_dir)
    end

    # Wait for the root command to complete
    use_MPI && MPI.Barrier(cart_comm)

    (g_nx, g_ny) = global_grid
    (px, py) = proc_dims
    (cx, cy) = cart_coords

    ((nx != ny) || (px != py)) && error("Domain slices are only implemented for square domains on a square process grid.")

    # Middle row
    cy_mid = cld(py, 2) - 1
    if cy == cy_mid
        y_mid = cld(g_ny, 2) - ny * cy + 1
        output_file_path_X = build_file_path(params, file_name * "_X")
        open(output_file_path_X, "w") do file
            col_range = y_mid:y_mid
            row_range = 1:nx
            write_data_to_file(params, data, col_range, row_range, file; for_3D=false)
        end
    end

    # Middle column
    cx_mid = cld(px, 2) - 1
    if cx == cx_mid
        x_mid = cld(g_nx, 2) - nx * cx + 1
        output_file_path_Y = build_file_path(params, file_name * "_Y")
        open(output_file_path_Y, "w") do file
            col_range = 1:ny
            row_range = x_mid:x_mid
            write_data_to_file(params, data, col_range, row_range, file; for_3D=false)
        end
    end

    # Diagonal
    if cx == cy
        output_file_path_D = build_file_path(params, file_name * "_D")
        open(output_file_path_D, "w") do file
            col_range = 1:1
            row_range = params.ideb:(params.row_length+1):(params.ifin+params.row_length+1)
            write_data_to_file(params, data, col_range, row_range, file; for_3D=false, direct_indexing=true)
        end
    end

    if !no_msg && is_root && silent < 2
        if params.use_MPI
            println("Wrote slices to files $(joinpath(output_dir, file_name))_*_*x*")
        else
            println("Wrote slices to files $(joinpath(output_dir, file_name))_*")
        end
    end
end


function read_data_from_file(params::ArmonParameters{T}, data::ArmonDataOrDual,
        col_range, row_range, file; direct_indexing=false) where T
    @indexing_vars(params)

    vars_to_read = saved_variables(data isa ArmonDualData ? host(data) : data)

    for j in col_range
        for i in row_range
            if direct_indexing
                idx = i + j - 1
            else
                idx = @i(i, j)
            end

            for var in vars_to_read[1:end-1]
                var[idx] = parse(T, readuntil(file, ','))
            end
            vars_to_read[end][idx] = parse(T, readuntil(file, '\n'))
        end
    end
end


function read_sub_domain_file!(params::ArmonParameters, data::ArmonDataOrDual, file_name::String)
    (; nx, ny, nghost) = params

    file_path = build_file_path(params, file_name)

    open(file_path, "r") do file
        col_range = 1:ny
        row_range = 1:nx
        if params.write_ghosts
            col_range = inflate(col_range, nghost)
            row_range = inflate(row_range, nghost)
        end

        read_data_from_file(params, data, col_range, row_range, file)
    end
end

#
# Comparison functions
#

function compare_data(label::String, params::ArmonParameters,
        ref_data::ArmonData{V}, our_data::ArmonData{V}; mask=nothing) where V
    (; row_length, nghost, nbcell, comparison_tolerance) = params
    different = false

    for name in saved_variables()
        ref_val = getfield(ref_data, name)
        our_val = getfield(our_data, name)

        diff_mask = .~ isapprox.(ref_val, our_val; atol=comparison_tolerance)
        !params.write_ghosts && (diff_mask .*= our_data.domain_mask)
        !isnothing(mask) && (diff_mask .*= mask)
        diff_count = sum(diff_mask)

        if diff_count > 0
            !different && println("At $label:")
            different = true
            print("$diff_count differences found in $name")
            if diff_count ≤ 200
                println(" (ref ≢ current)")
                for idx in 1:nbcell
                    !diff_mask[idx] && continue
                    i, j = ((idx-1) % row_length) + 1 - nghost, ((idx-1) ÷ row_length) + 1 - nghost
                    @printf(" - %5d (%3d,%3d): %10.5g ≢ %10.5g (%10.5g)\n", idx, i, j, 
                        ref_val[idx], our_val[idx], ref_val[idx] - our_val[idx])
                end
            else
                println()
            end
        end
    end

    return different
end


function domain_mask_with_ghosts(params::ArmonParameters{T}, mask::V) where {T, V <: AbstractArray{T}}
    (; nbcell, nx, ny, nghost, row_length) = params

    r = nghost
    axis = params.current_axis

    for i in 1:nbcell
        ix = ((i-1) % row_length) - nghost
        iy = ((i-1) ÷ row_length) - nghost

        mask[i] = (
               (-r ≤ ix < nx+r && -r ≤ iy < ny+r)  # The sub-domain region plus a ring of ghost cells...
            && ( 0 ≤ ix < nx   ||  0 ≤ iy < ny  )  # ...while excluding the corners of the sub-domain...
            &&((axis == X_axis &&  0 ≤ iy < ny  )  # ...and excluding the ghost cells outside of the
            || (axis == Y_axis &&  0 ≤ ix < nx  )) # current axis
        ) ? 1 : 0
    end
end


function compare_with_file(params::ArmonParameters, data::ArmonData,
        file_name::String, label::String)

    ref_data = ArmonData(params)
    read_sub_domain_file!(params, ref_data, file_name)

    if params.use_MPI && params.write_ghosts
        domain_mask_with_ghosts(params, ref_data.domain_mask)
        different = compare_data(label, params, ref_data, data; mask=ref_data.domain_mask)
    else
        different = compare_data(label, params, ref_data, data)
    end

    if params.use_MPI
        different = MPI.Allreduce(different, |, params.cart_comm)
    end

    return different
end


function step_checkpoint(params::ArmonParameters, data::ArmonDualData, step_label::String)
    if params.compare
        wait(params)

        device_to_host!(data)
        h_data = host(data)

        step_file_name = params.output_file * @sprintf("_%03d_%s", params.cycle, step_label)
        step_file_name *= isnothing(params.axis) ? "" : "_" * string(params.axis)[1:1]

        if params.is_ref
            write_sub_domain_file(params, h_data, step_file_name; no_msg=true)
        else
            different = compare_with_file(params, h_data, step_file_name, step_label)
            if different
                write_sub_domain_file(params, h_data, step_file_name * "_diff"; no_msg=true)
                println("Difference file written to $(step_file_name)_diff")
            end
            return different
        end
    end

    return false
end
