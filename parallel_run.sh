echo "Initializing database.."
python -c "import optuna; optuna.create_study(storage='sqlite:///calibration.db', study_name='calibration', load_if_exists=True)"
echo "Database initialized"
for i in {1..8}; do
    echo "Launching Worker $i..."
    nice -n 10 python -u calibrate.py > worker_${i}.log 2>&1 &
done
# keep screen alive
wait
