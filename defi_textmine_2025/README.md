# Community Prediction Competition | Défi TextMine 2025

* code: defi-text-mine-2025
* name: Défi TextMine 2025 Extraction de relations pour l'analyse des rapports de renseignement
* description: the challenge focuses on the task of extracting relationships between textual entities whose are alredy recognized.
* page: https://www.kaggle.com/competitions/defi-text-mine-2025
* data: Text data, Information Extraction,  Relation Extraction
* scores: 
    - [Macro F1 (average F1 score for each relationship type)](https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/#989c)

## Setup

### VS Code settings
````json
{
    "workbench.colorTheme": "Default Dark Modern",
    "extensions.autoCheckUpdates": false,
    "extensions.autoUpdate": false,
    "update.enableWindowsBackgroundUpdates": false,
    "update.showReleaseNotes": false,
    "update.mode": "none",
    "telemetry.telemetryLevel": "off",
    "settingsSync.keybindingsPerPlatform": false,
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "python.terminal.launchArgs": [
        "-m",
        "IPython",
        "--no-autoindent"
    ],
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "remote.autoForwardPortsSource": "hybrid",
    "jupyter.askForKernelRestart": false,
    "flake8.args": [
        "--max-line-length=88"
    ],
    "git.suggestSmartCommit": false,
    "notebook.lineNumbers": "on",
}
````

### Create and activate a python (>=3.10,<3.12) environment

* Use anything you want: pyenv, conda, venv, ...
* install the dependencies: `pip install -r requirements.txt`

### Use kaggle API to get the data and submit your solution

#### Install and configure the kaggle API client package

* install: `pip install -U kaggle`
* Read the instruction [here](https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#api-credentials) to create your API credentials and set them up at `~/.kaggle/kaggle.json`
* Verify that everything works well: `kaggle competitions list --category "playground"`, you should see something like this:

````
ref                                                         deadline             category    reward  teamCount  userHasEntered  
----------------------------------------------------------  -------------------  ----------  ------  ---------  --------------  
https://www.kaggle.com/competitions/playground-series-s4e4  2024-04-30 23:59:00  Playground    Swag        732            True
````

#### Download data

* Competition data
````shell
challenge="defi-text-mine-2025"
mkdir -p data/${challenge}/raw
kaggle competitions download -c ${challenge} -p data
unzip data/${challenge}.zip -d data/${challenge}/raw
rm data/${challenge}.zip
````

#### Submission
```shell
kaggle competitions submit -c defi-text-mine-2025 -f data/defi-text-mine-2025/output/submission.csv -m "Message"
```

### _Weights & Bias_ to track experiment

* Create the project space: `https://wandb.ai/new-project?entityName={YOUR WANDB USERNAME}`
    - create a project with the challenge code `defi-text-mine-2025`
* install the wandb python package: `$ pip install wandb`
* login once in the terminal: `$ wandb login`
    - Get the API key from https://wandb.ai/authorize

## Tips

### Add special tokens to a Transformer tokenizer
Here we add 4 html opening and closing tags to mar the mentions of the first and second entities of the relation

````python
tokenizer.add_tokens(['<e1>', '</e1>', '<e2>', '</e2>'], special_tokens=True)
````