#!/bin/bash
#SBATCH --job-name=multi_target33
#SBATCH --array=0-16
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/output/target_%A_%a.out
#SBATCH --error=/scratch/output/target_%A_%a.err



cd $SLURM_SUBMIT_DIR
source ~/anaconda3/etc/profile.d/conda.sh
conda activate H100-unlearn


targets=("golf ball" "parachute" "church" "french horn" "chain saw" "gas pump" "candle" "mountain bike" "racket" "school bus" "spider web" "starfish" "Monet" "Leonardo Da Vinci" "Pablo Picasso" "Salvador Dali" "Van Gogh")


target="${targets[$SLURM_ARRAY_TASK_ID]}"
top_ratios=(0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8)




python -m revive.wanda --target="$target" --skill_ratio 0.02
python -m revive.save_union_over_time --target="$target" --timesteps 10 --skill_ratio 0.02
python -m revive.read_weigts --target="$target"
python -m revive.matrix_completion_lterative_Soft-Thresholded_SVD_gpu --target="$target"


for top_ratio in "${top_ratios[@]}"; do
    echo "Running wanda.top_ratio_csv for $target with ratio $top_ratio"
    python -m revive.top_k_sign_retention --target="$target" --top_ratio="$top_ratio"
done

python -m revive.neuron_max_scaling --target="$target" --csv_folder "/scratch/concept_revival/${target}/top_ratio_output"

cd benchmarking

artists=("Monet" "Leonardo Da Vinci" "Pablo Picasso" "Salvador Dali" "Van Gogh")


for top_ratio in "${top_ratios[@]}"; do
    echo "Running target: $target with top_ratio: $top_ratio"
    path="/scratch/recovered_models/${target}/Recover_via_500it_1e5cv_rankNone_ft64_Top${top_ratio}_Sign_Max_processing/model/filled_with_mag_seed2021.pt"


    is_artist=false
    for artist in "${artists[@]}"; do
        if [[ "$artist" == "$target" ]]; then
            is_artist=true
            break
        fi
    done

    if [[ "$is_artist" == true ]]; then
        echo "Running artist_erasure for artist target: $target"
        python -m artist_erasure --target="$target" --baseline concept-prune --ckpt_name="$path" --top_ratio="$top_ratio"
    else
        echo "Running object_erase for object target: $target"
        python -m object_erase --target="$target" --baseline concept-prune --removal_mode erase --ckpt_name="$path" --top_ratio="$top_ratio"
    fi
done
