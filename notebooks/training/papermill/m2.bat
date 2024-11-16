call "../../../.venv/Scripts/activate"
papermill "../raw/m2.ipynb" "../completed/m2_%1_cv%2.ipynb" ^
--cwd "../../../" --autosave-cell-every 0 --log-output --no-progress-bar ^
-p notebook_classification %1 ^
-p notebook_cv %2