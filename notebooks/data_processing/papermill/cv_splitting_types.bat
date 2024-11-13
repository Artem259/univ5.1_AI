call "../../../.venv/Scripts/activate"
papermill "../raw/cv_splitting.ipynb" "../completed/cv_splitting_types.ipynb" ^
--cwd "../../../" --autosave-cell-every 10 --log-output --no-progress-bar ^
-p classification "types"