call "../../../.venv/Scripts/activate"
papermill "../raw/m1.ipynb" "../completed/m1_models_cv1.ipynb" ^
--cwd "../../../" --autosave-cell-every 0 --log-output --no-progress-bar ^
-p notebook_classification %1 ^
-p notebook_cv %2