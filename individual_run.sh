#!/bin/bash

echo "Initializing databases.."
#python -c "import optuna; optuna.create_study(storage='sqlite:///Control_hab.db', study_name='Control_pre', load_if_exists=True)"
python -c "import optuna; optuna.create_study(storage='sqlite:///Resilient_hab.db', study_name='Resilient_pre', load_if_exists=True)"
#python -c "import optuna; optuna.create_study(storage='sqlite:///Susceptible_hab.db', study_name='Susceptible_pre', load_if_exists=True)"
echo "Databases initialized"

echo "Launching workers..."

# Mouse A: Resilient (3 Workers)
for i in $(seq 1 2);
do
	nice -n 10 python fit_individual.py --mouse Resilient > log_Res_${i}.txt 2>&1 &
done
#nice -n 10 python fit_individual.py --mouse Resilient > log_Res_2.txt 2>&1 &
#nice -n 10 python fit_individual.py --mouse Resilient > log_Res_3.txt 2>&1 &

# Mouse B: Susceptible (3 Workers)
#nice -n 10 python fit_individual.py --mouse Susceptible > log_Sus_1.txt 2>&1 &
#nice -n 10 python fit_individual.py --mouse Susceptible > log_Sus_2.txt 2>&1 &
#nice -n 10 python fit_individual.py --mouse Susceptible > log_Sus_3.txt 2>&1 &

# Mouse C: Control (2 Workers)
#nice -n 10 python fit_individual.py --mouse Control > log_Con_1.txt 2>&1 &
#nice -n 10 python fit_individual.py --mouse Control > log_Con_2.txt 2>&1 &

echo "All 2 workers launched in the background!"
wait
echo "Fitting completed"
