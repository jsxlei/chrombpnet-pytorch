# Dec 6th, 2025
## Fix
- 1.The dataloader crop_revcom_data for non-training data 

## Feature
- 1. The training data will be saved in cached_dir for reproducing
- 2. The ChromBPNetWrapper now is cleaner and consists of model and bias, use self.model for running shap and variant scoring

## Note
- 1. The model trained previouse might not be compatible with current version.
 
