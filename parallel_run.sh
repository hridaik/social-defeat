for i in {1..8}; do
    echo "Launching Worker $i..."
    # "nice -n 10" prevents SSH from lagging
    # "2>&1" captures CRASHES into the log file
    nice -n 10 python -u optuna_fit.py > worker_${i}.log 2>&1 &
done
# keep screen alive
wait
