
PROJECT_DIR = .
TEST_DIR = .
PERF_REF_DIR = $(TEST_DIR)/performance_data
DATA_REF_DIR = $(TEST_DIR)/reference_data

JULIA_THREADS ?= auto
JULIA ?= julia
JULIA_ARGS ?= -O3 --check-bounds=no --project=$(PROJECT_DIR) -t $(JULIA_THREADS)

MPI_PROCS ?= 25
MPI_EXEC ?= mpiexec
MPI_EXEC_ARGS ?= -n $(MPI_PROCS)
MPI_JL_ARGS ?= --color=yes

main_test_file = $(TEST_DIR)/main.jl
create_perf_ref_file = $(PERF_REF_DIR)/create_performance_reference.jl
create_data_ref_file = $(DATA_REF_DIR)/create_references.jl


# When target is 'test', forward other args to Julia
ifeq (test,$(firstword $(MAKECMDGOALS)))
  TEST_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(TEST_ARGS):;@:)
endif


.PHONY: create_perf_reference create_solver_reference short perf mpi all

create_perf_reference:
	$(JULIA) $(JULIA_ARGS) $(create_perf_ref_file)

create_solver_reference:
	$(JULIA) $(JULIA_ARGS) $(create_data_ref_file)

mpi:
	$(MPI_EXEC) $(MPI_EXEC_ARGS) $(JULIA) $(JULIA_ARGS) $(MPI_JL_ARGS) $(main_test_file) mpi

test:
	$(JULIA) $(JULIA_ARGS) $(main_test_file) $(TEST_ARGS)

all:
	$(JULIA) $(JULIA_ARGS) $(main_test_file) all
