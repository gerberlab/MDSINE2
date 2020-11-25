# Parse data into `mdsine2.Study` objects
```python
import mdsine2 as md2
import pandas as pd
```
## Required files
---
There are 4 separate tables that are needed to run the MDSINE2 model
   
#### 1) Taxonomy table
The taxonomy table, which we will refer to as `taxonomy`, is a `pd.DataFrame` object where the columns are `name`, `sequence`, `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, and `species`. An example is below:

| name  | sequence | kingdom | phylum | class | order | family | genus | species |
|:-----:|:--------:|:-------:|:------:|:-----:|:-----:|:------:|:-----:|:-------:|
| ASV_1 |AAAAAA| Bacteria | Bacteroidetes | Bacteroidia | Bacteroidales | Bacteroidaceae | Bacteroides | dorei/vulgatus |
| ASV_2 |TTTTTT| Bacteria | Verrucomicrobia | Verrucomicrobiae | Verrucomicrobiales | Verrucomicrobiaceae | Akkermansia | muciniphila |


#### 2) Count table
The count table, which we will refer to as `counts`, is a `pd.DataFrame` object  where the index are the names of the ASVs and the columns are the labels of the samples. This is the output from DADA2. An example is below

| name  | sample1 | sample2 | sample3 | sample4 |
| ----- | ------ | ------ | ------ | ------ |
| ASV_1 | 1234   | 4567   | 3383 | 3983 |
| ASV_2 | 9876   | 5432   | 1111 | 2222 |

#### 3) qPCR table
The qPCR table, which we will refer to as `qpcr`, is a `pd.DataFrame` object  where the index are the names of the samples and the columns dont matter. An example is below

|   | mass1 | mass2 | mass3 |
| ----- | ------ | ------ | ------ |
| sample1 | 11111   | 22222   | 33333 |
| sample2 | 44444   | 55555   | 66666 |
| sample3 | 77777   | 88888   | 99999 |
| sample4 | 12345   | 6789   | 1010101 |

This table assumes that there are 3 qPCR samples for each sample. There can b e any number of columns

#### 4) Meta-data table
The meta-data table, which we will refer to as `meta`, is a `pd.DataFrame` object  where the index are the names of the samples and the columns must include `time` and `subject`. 
`time` (`float`) is the time-point that the sample takes place and `subject` (`str`) is the name of the subject. If there are perturbations in the study, then the prefix of the column must be `perturbation:`. If the perturbation is active during that time point, then set the value to 1, otherwise to 0. An example is below

|   | time | subject | perturbation:pert1 | perturbation:pert2 |
| ----- | ------ | ------ | ------ | ------ |
| sample1 | 0   | subj1   | 0 | 1 |
| sample2 | 1   | subj1   | 1 | 0 |
| sample3 | 2   | subj2   | 1 | 0 |
| sample3 | 3   | subj3   | 0 | 0 |

In the above example, sample `sample1` is at time `0.0`, for subject `subj1` and the perturbation `pert2` is active.

## Code
---

To parse your own data, first create a `md2.TaxaSet` object:
```python
taxas = md2.TaxaSet(taxonomy_table=taxonomy)
```
Then, create the `md2.Study` object
```python
study = md2.Study(taxas=taxas)
study.parse(
    metadata=metadata,
    reads=reads,
    qpcr=qpcr)
```