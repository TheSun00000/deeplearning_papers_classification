# deeplearning_papers_classification

The goal of this project is to help deep learning researchers and enthusiasts organize their downloaded scientific papers. The script  scans the abstract (or introduction, header, etc ...) of the paper and then pass it through the machine learning model in order to classify the paper to the following classes:

 1. Computer Vision.
 2. NLP.
 3. Self supervised learning.
 4. Reinforcement learning.
 5. Other.


# Dataset:
The used dataset [paper_abs_task.csv](https://github.com/TheSun00000/deeplearning_papers_classification/blob/main/scraping/paper_abs_task.csv "paper_abs_task.csv") has been collected by us from the website: [paper with code](https://paperswithcode.com/). The collected dataset consists of pairs of ( Task, Abstract):
| Id | Task | Abstract |
|--|--|--|
| 0 | Task0 | Abstract0 |
| 1 | Task1 | Abstract1 |
| ... | ... | ... |
| 1295 | Task1295 | Abstract1295 |

such as Task_i is the class of the paper (CV, NLP, SSL, RL, Other).
***NOTE: We don't claim the ownership of any of the previous dataset and it won't be used in any kind of financial purposes !!!*** 
