pip install --no-cache-dir -r requirements.txt
cd dataset_manager
python dataset_manager_script.py
python dataset_manager_api.py
cd ..
cd neural_net
python nn_script.py
python nn_api.py
cd ..
cd frontend
python frontend_api.py
cd ..