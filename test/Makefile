JULIA?=julia
julia_cmd := $(JULIA) --color=yes --check-bounds=yes
srun_cmd := srun --pty -p gpu --gres=gpu:1 --mem=8000

default:
	@echo "Target List"
	@echo "  test      - run local test"
	@echo "  test-ext  - run local test with native extension"
	@echo "  test-gpu  - run local test with CUDA backend"
	@echo "  test-srun - run test with CUDA backend on server"

test:
	$(julia_cmd) runtests.jl

test-ext:
	MOCHA_USE_NATIVE_EXT=true $(julia_cmd) runtests.jl

test-gpu:
	MOCHA_USE_CUDA=true $(julia_cmd) runtests.jl

test-srun:
	$(srun_cmd) bash -c "MOCHA_USE_CUDA=true $(julia_cmd) runtests.jl"
