"""

ARGUMENT-MINING CODE

Tips On Managing This Project:
- Make sure you are running this project with a virtual environment! Use "virtualenv" OR "pipenv"! (I recommend using pipenv...)
    - pip install virtualenv
    - pip install pipenv
    - Look online on how to use these virtual environment managers!
- Make it clear between your IMPORTS and CODE! (Just like how I made those comment blocks...)
- Make sure you categorize all import statements correctly (ex: Standard, Third-Party, and Local)
- Make sure each category of imports is alphabetized for easier lookup
- Each package should only contain classes or functions of code. Main code should be run here in "__main__.py"
- If you want to add a new functionality (or package) to the project, make sure you add a "__init__.py" to make it a package!
- Use feature branches to modify different packages
    - i.e. if you want to add or update to the "data/" package, do that in the "data" branch
- There are 2 ways to run this project...
    1) Make sure you are at the top level of this project and type this command into the terminal:
        python -m argument-mining
    2) If you are editing this project from an IDE, I suggest you run the "run.py" file and it should work the same
- Make sure you are updating your dependency files often!
    - requirements.txt and test_requirements.txt (if you are using "virtualenv")
    - Pipfile and Pipfile.lock (if you are using "pipenv")
    
"""

###########
# IMPORTS #
###########

# Standard Library Imports
import os

# Third Party Library Imports


# Local Library Imports
from .data import data_cleaning


########
# CODE #
########

def main():
    # Clean Doccano Data By Splitting Data Into Best Labels and Conflict Labels
    dataset_path = os.path.join('argument_mining', 'resources', 'reu_argument-mining_dataset')
    # print(dataset_path)
    data_cleaning.clean(dataset_path)  # Path Of Dataset

    # Todo: Run Model Functions Here!


if __name__ == '__main__':
    main()
