#!/bin/bash

echo "Initializing databases.."
# {mouse_name}_{model}.db
# avg, avg, chd
python -c "import optuna; optuna.create_study(storage='sqlite:///noT_avg.db', study_name='noT_avg', load_if_exists=True)"
python -c "import optuna; optuna.create_study(storage='sqlite:///noD_avg.db', study_name='noD_avg', load_if_exists=True)"
python -c "import optuna; optuna.create_study(storage='sqlite:///M_avg.db', study_name='M_avg', load_if_exists=True)"
python -c "import optuna; optuna.create_study(storage='sqlite:///full_avg.db', study_name='full_avg', load_if_exists=True)"
echo "Databases initialized"

echo "Launching workers..."

# No T Model (2 Workers)
# for i in $(seq 1 6);
# do
# 	nice -n 10 python fit_after.py > log_Sus_${i}.txt 2>&1 &
# done
nice -n 10 python fit_reduced.py --mouse avg --model noT > log_noT_2.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse avg --model noT > log_noT_3.txt 2>&1 &

# No D Model (2 Workers)
nice -n 10 python fit_reduced.py --mouse avg --model noD > log_noD_1.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse avg --model noD > log_noD_2.txt 2>&1 &

# Only M Model (2 Workers)
nice -n 10 python fit_reduced.py --mouse avg --model M_only > log_M_1.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse avg --model M_only > log_M_2.txt 2>&1 &

# Full Model (2 Workers)
nice -n 10 python fit_reduced.py --mouse avg --model full > log_Full_1.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse avg --model full > log_Full_2.txt 2>&1 &


echo "All workers launched in the background!"
wait
echo "Fitting completed"
