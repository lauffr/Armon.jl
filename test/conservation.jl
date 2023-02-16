
import Armon: @i, @indexing_vars, init_test, time_loop, conservation_vars


@testset "Conservation" begin
    @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ)
        ref_params = get_reference_params(test, Float64; maxcycle=10000, maxtime=10000)

        data = ArmonDualData(ref_params)
        init_test(ref_params, data)

        init_mass, init_energy = conservation_vars(ref_params, data)
        time_loop(ref_params, data)
        end_mass, end_energy = conservation_vars(ref_params, data)

        @test   init_mass ≈ end_mass    atol=1e-12
        @test init_energy ≈ end_energy  atol=1e-12
    end
end
