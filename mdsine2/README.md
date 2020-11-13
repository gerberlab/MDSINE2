# Install dependencies (Python 3.7.3)
* [pylab](https://github.com/gerberlab/PyLab) == 3.0.1
* biopython==1.76
* scikit-bio==0.5.6
* ete3

# Running the model
Run the MDSINE2 model with the Gibson dataset
## Parse the data
```python
python make_real_subjset.py
```

## Fit the Negative Binomial Dispersion parameters
```python
python main_negbin.py \
    --data-seed 0 \
    --basepath output_negbin/ \
    --n-samples 10000 --burnin 4000
    --param-filename tmp/negbin_params.tsv
```

## Fit the qPCR measurements
```python
python main_qpcr.py \
    --data-filename pickles/real_subjectset.pkl \
    --output-basepath output_qpcr/
    --param-filename tmp/qpcr_param.tsv
```

## Run the Model

These commands run and save the model, plots the posteriors, and runs validation (if possible)

#### Healthy cohort
```python
python main_real.py \
    --dataset gibson \
    --data-seed 0 \
    --init-seed 0 \
    --basepath output/ \
    --n-samples 15000 \
    --burnin 5000 \
    --ckpt 100 \
    --healthy 1
```
#### Ulcerative colitis cohort
```python
python main_real.py \
    --dataset gibson \
    --data-seed 0 \
    --init-seed 0 \
    --basepath output/ \
    --n-samples 15000 \
    --burnin 5000 \
    --ckpt 100 \
    --healthy 0
```
#### Run cross validation and leave out the first subject for validation
```python
python main_real.py \
    --dataset gibson \
    --data-seed 0 \
    --init-seed 0 \
    --basepath output/ \
    --n-samples 15000 \
    --burnin 5000 \
    --ckpt 100 \
    --healthy 0 \
    --leave-out 0
```
#### Dispatch leave-one-out cross validation into lsf jobs (each left out subject is a new job)
```python
python main_real.py \
    --dataset gibson \
    --data-seed 0 \
    --init-seed 0 \
    --basepath output/ \
    --n-samples 15000 \
    --burnin 5000 \
    --ckpt 100 \
    --healthy 0 \
    --cross-validate 1 \
    --use-bsub 1
```