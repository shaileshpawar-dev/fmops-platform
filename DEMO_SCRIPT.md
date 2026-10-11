# Demonstration Script for Faculty

This script outlines the exact steps to reproduce both training workflows on the FMOps Platform.

## Demo A: Train Inside FMOps
1. Go to the **New ML Project** page.
2. Under **1. Upload training dataset**, upload `tests/data/loan_default.csv`.
3. Wait for the dataset to validate and display the profile.
4. Set **Target Column** to `default` and **Negative Class** to `0`.
5. Click **Start AutoML Training**.
6. Wait for the pipeline to finish and click the suggested model version.
7. Click **Promote to Staging**, then **Promote to Production** (using the dashboard or CLI).
8. Go to the model deployment view and send a test prediction.

## Demo B: Direct Pretrained Model Upload
1. Ensure you have the `demo_model.joblib` file ready (generated locally using a small python script).
2. Go to the **Models** page.
3. Click **Upload Pretrained Model**.
4. Set Model Name to `demo_classifier`.
5. Select the `demo_model.joblib` file.
6. Provide standard evaluation metrics (default JSON is populated in the form) - Note: These are recorded as unverified client metadata.
7. Provide the expected Input Signature (default JSON is populated in the form).
8. Click **Upload**.
9. Open the newly uploaded model version.
10. Note that since the model lacks verified evaluation metrics, it will not pass the automated approval gate. To deploy it, an evaluation workflow on a labeled dataset is required, or manual override by an admin.
11. Submit a test prediction from the deployment pane (once evaluated and deployed).

**Notes on Railway Deployment:**
Since Railway is configured without a persistent volume in this setup, all data (models, metrics, datasets) will reset on the next push or container restart. This allows for a clean slate for the presentation.
