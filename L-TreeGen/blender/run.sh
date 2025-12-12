# Record the start time and convert to date and time
start_time=$(date +%s)
start_date=$(date)

python3 scripts/distributed.py \
  --num_gpus 1 \
  --workers_per_gpu 8 \
  --code single \
  --input_models_path demo_output/interpolation_num-new-tree-10/rotated_obj/tree_obj_filepaths.json \
  --output_dir rendering_results

# Record the end time and convert to date and time
end_time=$(date +%s)
end_date=$(date)

# Calculate the total duration
elapsed=$((end_time - start_time))

# Convert the total duration to hours, minutes, and seconds
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60))
seconds=$((elapsed % 60))

# Print the execution results
echo "Start time: $start_date"
echo "End time: $end_date"
echo -n "Total time elapsed: "

# Print only if hours, minutes, or seconds are not zero
if [ $hours -gt 0 ]; then
  echo -n "$hours hours "
fi

if [ $minutes -gt 0 ] || [ $hours -gt 0 ]; then # Display minutes if there are hours
  echo -n "$minutes minutes "
fi

if [ $seconds -gt 0 ] || [ $minutes -gt 0 ] || [ $hours -gt 0 ]; then # Display seconds if there are minutes
  echo "$seconds seconds"
fi

echo ""