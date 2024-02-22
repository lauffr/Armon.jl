
@testset "Type stability" begin
    @testset "Limiters" begin
        for type in (Float32, Float64)
            x = type(0.456)
            @test type == typeof(@inferred Armon.limiter(x, Armon.NoLimiter()))
            @test type == typeof(@inferred Armon.limiter(x, Armon.MinmodLimiter()))
            @test type == typeof(@inferred Armon.limiter(x, Armon.SuperbeeLimiter()))
        end
    end
end
