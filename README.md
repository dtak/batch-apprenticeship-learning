### Truly Batch Apprenticeship Learning with Deep Sucessor Features

<b>The supplementary material (Appendix) to the conference submission made to IJCAI-2019 can be found in this repository in this link : https://github.com/dtak/batch-apprenticeship-learning/blob/master/paper_submission/ijcai_appendix_dsfn.</b>


This repository contains a collection of methods for batch, off-policy inverse reinforcement learning under unknown dynamics. Test environments were OpenAI Gyms (Mountaincar-v0, Acrobot-v1, Cartpole-v0) and Sepsis (MIMIC-III). The methods listed are DSFN, TRIL, LSTD-mu, and SCIRL. The dependencies are python 3.5+, tensorflow 1.4+, OpenAI baselines, Scikit-learn 0.19+ and Numpy 1.14+.

- [Link](http://example.net)

```
# Classical Control
python src/main.py TASK_NAME --model_id MODEL_ID --e_filename DATA_PATH
# alternatively use src/run.py

# Sepsis
python src/run_sepsis.py sepsis --model_id MODEL_ID --e_filename DATA_PATH
```

See the default arg settings in `main.py`. See the experimental details in `task/`.


