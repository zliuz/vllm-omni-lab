.PHONY: check-env setup clone download

check-env:
	bash scripts/check_env.sh

setup:
	bash scripts/setup_env.sh

clone:
	bash scripts/clone_upstream.sh

download:
	bash scripts/download_models.sh
