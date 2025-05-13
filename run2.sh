#!/bin/bash

# Function to generate a unique seed
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

# Base log directory
BASE_LOGDIR=~/logdir_teacher_student

# List of configs to run
CONFIGS=(
  # "cartpole_small"
  # "gymnasium_bandit5"
  # "gymnasium_lavatrail8"
  # "gymnasium_blindpick"
  # "gymnasium_lavatrail8_imitationlatent"
  # "gymnasium_lavatrail8_studentteacherlatent"
  # "gymnasium_lavatrail8_teacherstudentlatent"

  # "gymnasium_lavatrail8_imitationlatent_0.1"
  # "gymnasium_lavatrail8_imitationlatent_1"
  # "gymnasium_lavatrail8_imitationlatent_10"

  # "gymnasium_lavatrail8_studentteacherlatent_0.1"
  # "gymnasium_lavatrail8_studentteacherlatent_1"
  # "gymnasium_lavatrail8_studentteacherlatent_10"

  # "gymnasium_lavatrail8_teacherstudentlatent_0.1"
  # "gymnasium_lavatrail8_teacherstudentlatent_1"
  # "gymnasium_lavatrail8_teacherstudentlatent_10"
  # "gymnasium_lavatrail8_teacherstudentlatent_1"
  # "gymnasium_lavatrail8_teacherstudentlatent_2"
  "gymnasium_lavatrail8_teacherstudentlatent_4"

  # "gymnasium_lavatrail8_unprivileged"
)

NUM_SEEDS=
INITIAL_SEED=$(generate_unique_seed)

SEEDS=()
for ((i=0; i<$NUM_SEEDS; i++)); do
  SEEDS+=($((INITIAL_SEED + i)))
done

export MUJOCO_GL=egl;

# Iterate
for CONFIG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    LOGDIR="${BASE_LOGDIR}/${CONFIG}_${SEED}"

    echo "Running Teacher-Student with config ${CONFIG} and seed ${SEED}, logging to ${LOGDIR}"

    timeout 4h python3 -u thesis/teacher_student/train.py \
      --configs ${CONFIG} \
      --seed "$SEED"

    if [ $? -eq 124 ]; then
      echo "Command timed out for config ${CONFIG} and seed ${SEED}."
    else
      echo "Command completed for config ${CONFIG} and seed ${SEED}."
    fi

    echo "-----------------------"
  done
done

echo "All tasks complete." 