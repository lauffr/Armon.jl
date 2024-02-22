
@testset "Conservation" begin
    @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ)
        ref_params = get_reference_params(test, Float64; maxcycle=10000, maxtime=10000)

        data = BlockGrid(ref_params)
        Armon.init_test(ref_params, data)

        init_mass, init_energy = Armon.conservation_vars(ref_params, data)
        Armon.time_loop(ref_params, data)
        end_mass, end_energy = Armon.conservation_vars(ref_params, data)

        @test   init_mass ≈ end_mass    atol=1e-12
        @test init_energy ≈ end_energy  atol=1e-12
    end
end
