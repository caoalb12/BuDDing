# BuDDing

BuDDing is a project focused on implementing efficient inference with dynamic pruning and parallelization. The pipeline consists of 4 main steps:

## Step 1: Generate Omission Sets

Run every cell in order in `GenerateOmissionSets.ipynb`.

**Output:** Results are saved in `{dataset_name}_pruning_log.json` files for each of the three datasets.

## Step 2: Create Dataset to Train Router Model

Run every cell in order in `MakeRouterTrainingData.ipynb`.

**Output:** The router training dataset is saved in `router_training_data.jsonl`.

## Step 3: Train Router Model

Run `RouterTraining.ipynb` to train the router model.

**Output:** The saved router model is available in `best_model.pt`.

## Step 4: Evaluate and Implement

Run `EvaluateBuDDing.ipynb` cell by cell to evaluate the implementation.

### Step 4a: Batching with Dynamic Pruning
See output metrics in the notebook cell outputs.

### Step 4b: Parallelization
See output metrics in the notebook cell outputs.

**Output:** View the evaluation metrics directly in the notebook cell outputs.
