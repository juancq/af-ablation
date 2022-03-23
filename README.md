# Who gets catheter ablation treatment for atrial fibrillation in New South Wales?

This is the source code used to model who receives catheter ablation and early ablation for patients non-valvular atrial fibrillation (AF).

Please refer to the paper [An observational study of clinical and health system factors associated with catheter ablation and early ablation treatment for atrial fibrillation in Australia](https://www.medrxiv.org/content/10.1101/2021.09.03.21263104v1) for more details.

## Models
We used Cox proportional hazard models for determining the risk factors associated with receiving catheter ablation (dependent variable). The time-to-event of catheter ablation was determined by calculating the number of months from the index admission until the earliest event among 1) the first occurrence of ablation, 2) the end of the time-at-risk window (31 October 2018), and 3) death. Among patients who received an ablation, the risk factors associated with receiving early ablation vs late ablation was determined using logistic regression. 