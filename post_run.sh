#!/bin/bash

echo "Initializing databases.."
# {mouse_name}_{model}.db
# puc, avg, chd
python -c "import optuna; optuna.create_study(storage='sqlite:///noT_puc.db', study_name='noT_puc', load_if_exists=True)"
python -c "import optuna; optuna.create_study(storage='sqlite:///noD_puc.db', study_name='noD_puc', load_if_exists=True)"
python -c "import optuna; optuna.create_study(storage='sqlite:///M_puc.db', study_name='M_puc', load_if_exists=True)"
echo "Databases initialized"

echo "Launching workers..."

# No T Model (3 Workers)
# for i in $(seq 1 6);
# do
# 	nice -n 10 python fit_after.py > log_Sus_${i}.txt 2>&1 &
# done
nice -n 10 python fit_reduced.py --mouse puc --model noT > log_noT_2.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse puc --model noT > log_noT_3.txt 2>&1 &

# No D Model (3 Workers)
nice -n 10 python fit_reduced.py --mouse puc --model noD > log_noD_1.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse puc --model noD > log_noD_2.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse puc --model noD > log_noD_3.txt 2>&1 &

# Only M Model (2 Workers)
nice -n 10 python fit_reduced.py --mouse puc --model M_only > log_M_1.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse puc --model M_only > log_M_2.txt 2>&1 &

echo "All workers launched in the background!"
wait
echo "Fitting completed"
