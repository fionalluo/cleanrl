#!/bin/bash

# Generate a unique seed based on the current date and time
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

# Base log directory
BASE_LOGDIR=~/logdir_teacher_student

# List of configurations (customize as needed)
CONFIGS=(
  # "gymnasium_lavatrail8"
  # "gymnasium_blindpick"
  # "gymnasium_lavatrail8_imitationlatent"
  # "gymnasium_lavatrail8_studentteacherlatent"
  # "gymnasium_lavatrail8_teacherstudentlatent"

  # "gymnasium_lavatrail8_imitationlatent_0.1"
  # "gymnasium_lavatrail8_imitationlatent_1"
  "gymnasium_lavatrail8_imitationlatent_2"
  "gymnasium_lavatrail8_imitationlatent_4"
  # "gymnasium_lavatrail8_imitationlatent_10"

  # "gymnasium_lavatrail8_studentteacherlatent_0.1"
  # "gymnasium_lavatrail8_studentteacherlatent_1"
  # "gymnasium_lavatrail8_studentteacherlatent_2"
  # "gymnasium_lavatrail8_studentteacherlatent_4"
  # "gymnasium_lavatrail8_studentteacherlatent_10"

  # "gymnasium_lavatrail8_teacherstudentlatent_0.1"
  # "gymnasium_lavatrail8_teacherstudentlatent_1"
  # "gymnasium_lavatrail8_teacherstudentlatent_2"
  "gymnasium_lavatrail8_teacherstudentlatent_4"
  "gymnasium_lavatrail8_teacherstudentlatent_10"

  # "gymnasium_lavatrail8_unprivileged"
)

# Number of seeds to generate
NUM_SEEDS=4

# Generate the initial unique seed
INITIAL_SEED=$(generate_unique_seed)

# Generate the list of seeds
SEEDS=()
for ((i=0; i<$NUM_SEEDS; i++)); do
  SEEDS+=($((INITIAL_SEED + i)))
done

# Calculate the number of configurations and seeds
NUM_CONFIGS=${#CONFIGS[@]}
NUM_TOTAL_JOBS=$(($NUM_CONFIGS * $NUM_SEEDS))

echo "Submitting $NUM_TOTAL_JOBS jobs..."

# Loop through all config-seed combinations and submit jobs individually
for ((config_idx=0; config_idx<$NUM_CONFIGS; config_idx++)); do
  CONFIG="${CONFIGS[$config_idx]}"

  for ((seed_idx=0; seed_idx<$NUM_SEEDS; seed_idx++)); do
    SEED="${SEEDS[$seed_idx]}"

    # Generate the logdir for each config-seed pair
    LOGDIR="${BASE_LOGDIR}/${CONFIG}_${SEED}"

    # Create a temporary helper script for each job
    HELPER_SCRIPT="teacher_student_job_${config_idx}_${seed_idx}.sh"

    cat << EOF > $HELPER_SCRIPT
#!/usr/bin/env bash
## dj-partition settings
#SBATCH --job-name=teacher_student
#SBATCH --output=outputs/ts_%A_%a.out
#SBATCH --error=outputs/ts_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --mem=40G
#SBATCH --exclude=kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,dj-2080ti-0.grasp.maas
##SBATCH --nodelist=dj-l40-0.grasp.maas

export MUJOCO_GL=egl;

CONFIG=$CONFIG
SEED=$SEED
BASE_LOGDIR=$BASE_LOGDIR
LOGDIR="\$BASE_LOGDIR/\$CONFIG_\$SEED"

echo "Running Teacher-Student with config \$CONFIG and seed \$SEED, logging to \$LOGDIR"

timeout 12h python3 -u thesis/teacher_student/train.py \\
  --configs "\$CONFIG" \\
  --seed "\$SEED"

if [ \$? -eq 124 ]; then
  echo "Command timed out for config \$CONFIG and seed \$SEED."
else
  echo "Command completed for config \$CONFIG and seed \$SEED."
fi

echo "-----------------------"
EOF

    # Make the helper script executable
    chmod +x $HELPER_SCRIPT

    # Submit the job
    sbatch $HELPER_SCRIPT

    # Clean up
    rm $HELPER_SCRIPT
  done
done
