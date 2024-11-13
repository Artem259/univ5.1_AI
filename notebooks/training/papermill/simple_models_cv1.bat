call "../../../.venv/Scripts/activate"
papermill "../raw/simple.ipynb" "../completed/simple_models_cv1.ipynb" ^
--cwd "../../../" --autosave-cell-every 0 --log-output --no-progress-bar ^
-p notebook_classification "models" ^
-p notebook_cv 1