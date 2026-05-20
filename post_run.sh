#!/bin/bash

echo "Initializing databases.."
# {mouse_name}_{model}.db
# chd, chd, chd
python -c "import optuna; optuna.create_study(storage='sqlite:///noT_chd.db', study_name='noT_chd', load_if_exists=True)"
python -c "import optuna; optuna.create_study(storage='sqlite:///noD_chd.db', study_name='noD_chd', load_if_exists=True)"
python -c "import optuna; optuna.create_study(storage='sqlite:///M_only_chd.db', study_name='M_only_chd', load_if_exists=True)"
# python -c "import optuna; optuna.create_study(storage='sqlite:///full_chd.db', study_name='full_chd', load_if_exists=True)"
echo "Databases initialized"

echo "Launching workers..."

# No T Model (2 Workers)
# for i in $(seq 1 6);
# do
# 	nice -n 10 python fit_after.py > log_chd_Sus_${i}.txt 2>&1 &
# done
nice -n 10 python fit_reduced.py --mouse chd --model noT > log_chd_noT_1.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse chd --model noT > log_chd_noT_2.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse chd --model noT > log_chd_noT_3.txt 2>&1 &

# No D Model (2 Workers)
nice -n 10 python fit_reduced.py --mouse chd --model noD > log_chd_noD_1.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse chd --model noD > log_chd_noD_2.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse chd --model noD > log_chd_noD_3.txt 2>&1 &

# Only M Model (2 Workers)
nice -n 10 python fit_reduced.py --mouse chd --model M_only > log_chd_M_1.txt 2>&1 &
nice -n 10 python fit_reduced.py --mouse chd --model M_only > log_chd_M_2.txt 2>&1 &

# Full Model (2 Workers)
# nice -n 10 python fit_reduced.py --mouse chd --model full > log_chd_Full_1.txt 2>&1 &
# nice -n 10 python fit_reduced.py --mouse chd --model full > log_chd_Full_2.txt 2>&1 &


echo "All workers launched in the background!"
wait
echo "Fitting completed"
