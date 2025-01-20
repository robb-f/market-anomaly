# Market Anomaly Detection

## Description
A logistic regression ML model that predicts anomalies in the market based on data from the years 2000 to 2021.

First, exploratory data analysis (EDA) was conducted, and can be found in the `market-anomaly.ipynb` file. However, this
code is included in the `anomaly.py` file as well.

After the ML model is trained, the predictions are written back to the CSV file in `docs/output.csv`. The user then has the
option to have an investment strategy explained to them based on this [Qwen LLM](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).

## Usage
You should clone this repository first.
```nushell
    $ pip install -r requirements.txt
    $ python3 anomaly.py
```
On the first iteration, please allow approximately 5 minutes for the LLM to install locally. The LLM will take a little time to analyze
the date you provide.

## Contributing
Anyone is welcome to contribute to the project! Please review the issues and pull requests before submitting something.
* If you believe the model should be trained differently, please open a pull request. Please explain why you believe the model should be 
trained with the different parameters/hyperparameters you chose. Please tag the PR with the "model training" label.
* If you believe the model shouldn't be a logistic regression model, please open an issue, and explain why a different model should be used.
Proposed parameters are welcome, but not necessary. Please tag the issue with the "new model" label.
* If you encounter a bug and it is reproducible, please indicate how it can be reproduced. If you have a bug fix proposal, please submit it as
a PR. If you don't have a proposal, open an issue. For either case, you should tag the PR/issue with the "bug" label.
* If you'd like to see a new feature, please describe it in as much detail as possible. For suggestions on implementing the feature, you should
open a PR. Otherwise, just create a new issue. Please tag your submission with the "new feature" label.
* For questions about anything relating to this project, open an issue with the label "question".
* For anything else that doesn't fit anything in the above categories, open an issue/PR and tag it with the "misc" label.

------
*Last updated: Jan. 19, 2025*