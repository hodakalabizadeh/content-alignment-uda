#!/bin/bash
# Submit the whole pipeline for one config as chained Slurm jobs, each starting only if the jobs it
# needs succeeded:
#   train CA, one job per fold (in parallel) -> evaluate CA (all folds)
#                                            -> train downstream on the CA fold with the lowest validation
#                                               loss, one job per downstream seed -> evaluate downstream
#
# Usage, from the repository root with the Python environment activated:
#   bash scripts/slurm/run_all.sh <config> [run name, default: ca] [CA seed, default: 42] \
#        ["downstream seeds", default: the CA seed] ["extra CA training flags", default: none]
#   e.g.  bash scripts/slurm/run_all.sh obhc
#         bash scripts/slurm/run_all.sh obhc ca_s43 43 "42 43 44"
#         bash scripts/slurm/run_all.sh obhc ca_dir 42 "42 43 44" "--lambda-disc 0 --lambda-seg 0"   # ablation
set -euo pipefail
cfg=$1
run=${2:-ca}
seed=${3:-42}
downstream_seeds=${4:-$seed}
read -r -a ca_flags <<< "${5:-}"
folds=$(python -c "import yaml; print(yaml.safe_load(open('configs/$cfg.yaml'))['ca']['folds'])")
mkdir -p outputs/logs
submit() {  # submit <after job ids, colon-separated, or ""> <stage> <job-name suffix> [extra flags]
    sbatch --parsable --job-name="${cfg}_${run}_$2$3" ${1:+--dependency=afterok:$1} \
        "scripts/slurm/$2.sh" "$cfg" --run-name "$run" "${@:4}"
}
ca_jobs=()
for ((k = 0; k < folds; k++)); do
    ca_jobs+=("$(submit "" train_content_alignment "_f$k" --fold "$k" --seed "$seed" ${ca_flags[@]+"${ca_flags[@]}"})")
done
all_ca=$(IFS=:; echo "${ca_jobs[*]}")
j_eval_ca=$(submit "$all_ca" eval_content_alignment "" --seed "$seed")
echo "Submitted ${cfg}/${run}: train CA folds ${ca_jobs[*]} (seed $seed${5:+, flags: $5}) -> eval CA $j_eval_ca"
for s in $downstream_seeds; do
    j_ds=$(submit "$all_ca" train_downstream_seg "_s$s" --seed "$s")
    j_ev=$(submit "$j_ds" eval_downstream_seg "_s$s" --seed "$s")
    echo "  downstream seed $s: train $j_ds -> eval $j_ev"
done
echo "Logs: outputs/logs/${cfg}_${run}_<stage>-<job id>.out   Results: outputs/${cfg}/${run}/"
