# justfile
set shell := ["bash", "-euo", "pipefail", "-c"]

git-status:
	git status 2>&1 >| /tmp/just.log
	nvim -c "cfile /tmp/just.log | copen"

add-note MESSAGE:
    note="$$(date -u +%Y-%m-%dT%H:%M:%SZ)  {{MESSAGE}}"; \
    git notes --ref=ops append -m "$$note" || git notes --ref=ops add -f -m "$$note"; \
    git remote | grep -qx origin && git push -q origin refs/notes/ops || true

# example recipe + logging
git-branches-mtime:
    git for-each-ref --sort=-committerdate \
        --format='%(refname:short)  %(committerdate:iso8601)' \
        refs/heads refs/remotes
    just add-note "git-branches-mtime"

# justfile snippet ─ create and log a new branch
new-branch BRANCH:
    git switch -c {{BRANCH}} 2>/dev/null || git checkout -b {{BRANCH}}
    just add-note "new-branch {{BRANCH}}"

which-python:
	which python; which pip; echo $PWD;

dvc-commit:
	dvc commit --force

dvc-repro-s:
	export OC_CAUSE=1 && \
	export HYDRA_FULL_ERROR=1 && \
	echo "Reproducing with DVC..." && \
	dvc repro -s v1_ridge_optuna_trial_standard_scaler_2 --allow-missing --force

dvc-repro-st:
	export OC_CAUSE=1 && \
	export HYDRA_FULL_ERROR=1 && \
	echo "Reproducing troubleshooting with DVC..." && \
	dvc repro -s v1_ridge_optuna_trial --allow-missing --force | tee -a "${REPRO_LOG}" 2>&1

dvc-repro-st_2:
	export OC_CAUSE=1 && \
	export HYDRA_FULL_ERROR=1 && \
	echo "Reproducing troubleshooting with DVC..." && \
	dvc repro -s v1_ridge_optuna_trial_2 --allow-missing --force | tee -a "${REPRO_LOG}" 2>&1

tf-init:
	{ [ -d infra/.terraform ] && rm -rf infra/.terraform; \
	cd infra; \
	TF_IN_AUTOMATION=true terraform init -no-color; \
	} 2>&1 | tee /tmp/terraform_init.log
	pbcopy < /tmp/terraform_init.log

tf-plan:
	cd infra; \
	{ TF_IN_AUTOMATION=true terraform plan -no-color; \
	} 2>&1 | tee /tmp/terraform_plan.log; \
	pbcopy < /tmp/terraform_plan.log

tf-apply:
	cd infra; \
	{ TF_IN_AUTOMATION=true terraform apply -no-color; \
	} 2>&1 | tee /tmp/terraform_apply.log; \
	pbcopy < /tmp/terraform_apply.log

sim-inference:
	export OC_CAUSE=1 && \
	export HYDRA_FULL_ERROR=1 && \
	{ echo "OC_CAUSE: $OC_CAUSE"; \
	echo "HYDRA_FULL_ERROR: $HYDRA_FULL_ERROR"; \
	echo "Running inference simulation..."; \
	$CMD_PYTHON "$PROJECT_ROOT/bin/simulate_inference_mlflow.py"; \
	} 2>&1 | tee /tmp/sim-inference.log && \
	pbcopy < /tmp/sim-inference.log

backup-repo:
	parent_dir=$(dirname "$PROJECT_ROOT"); \
	repo="$(basename "$PROJECT_ROOT")"; \
	stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	backup="${repo}_backup_${stamp}"; \
	{ [[ -n ${PROJECT_ROOT:-} ]] || { echo "PROJECT_ROOT not set, exiting."; exit 1; } && \
	cd "$parent_dir" || exit 1; \
	echo "Running script in directory: $parent_dir"; \
	cp -a "$repo" "$backup"; \
	du -sh "$backup"; \
	} 2>&1 | tee /tmp/backup-repo.log && \
	echo "Backup completed successfully. Backup file: $backup"

restore-repo:
	parent_dir=$(dirname "$PROJECT_ROOT"); \
	repo="$(basename "$PROJECT_ROOT")"; \
	cd "$parent_dir" || exit 1; \
	rm -rf "$repo"; \
	mv "$backup" "$repo"
