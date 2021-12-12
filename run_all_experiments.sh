source venv/bin/activate
python3.9 run_experiment.py with random_config --name=random_config
python3.9 run_experiment.py with kernel_shap_config --name=kernel_shap_config
python3.9 run_experiment.py with lime_config --name=lime_config
python3.9 run_experiment.py with custom_hill_climber_tpn_config --name=custom_hill_climber_tpn_config
python3.9 run_experiment.py with custom_hill_climber_tps_config --name=custom_hill_climber_tps_config
