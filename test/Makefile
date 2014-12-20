julia_cmd := julia --color=yes --check-bounds
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
	bash -c "MOCHA_USE_NATIVE_EXT=true $(julia_cmd) runtests.jl"

test-gpu:
	bash -c "MOCHA_USE_CUDA=true $(julia_cmd) runtests.jl"

test-srun:
	$(srun_cmd) bash -c "$(julia_cmd) runtests.jl"
