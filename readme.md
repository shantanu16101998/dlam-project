## Activate virtual env
source .venv/bin/activate


## Requirements

Your `final_submission.zip` must contain:

```text
predict.py
requirements.txt
checkpoint.pt
src/  # optional
```

During private evaluation, instructors run:

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

It means we need to change predict.py to accomodate model

just run predict.py inside submission.


## Commands to run

python .\final_code\predict.py --input_dir .\data\input --output_file .\predictions.csv --checkpoint .\checkpoint.pt
python .\final_code\train.py
