# nnUNet (not a new UNet)

---
Author: Marc-Antoine Fortin, Mars 2025.
Email: marc.a.fortin@ntnu.no
Github: @mafortin

---

## Preface:

- In general, the [How to use nnUNet file](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md) is an extremely relevant source of information that I highly recommend to have in complementary with this SOP.
- For more niche questions, the [documentation folder](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation) is also extremely relevant.
- All of these commands have to be executed in your nnUNet virtual environment otherwise, the nnUNet functions won't be recognized.
	- A virtual Python environment can be easily created with conda/miniconda/pip. Check online for resources regarding those.

---

## General SOP for nnUNet

1) Rename/Reorganize your data to the nnunet naming convention as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) using the script `rename_files_nnunet_convention.py`.
2) Split your dataset into one Training and  one Test sets using the `train-test-split-datasets.py`. 
	- If you just want to produce the `.JSON` files for savng which subejcts is training and testing, you can use the flag `--skip-copy`, but you will have to move them after manually as described with the following step. If you do not use that flag, you can simply skip the next step.
3) If you already have `.JSON` files with the training and test subject split of your data or simply skipped the copy of the files when you created the `.JSON` files, move the Training and Testing subjects to their respective folders with the `reorganize_nii_files.py` script.
4) Make sure you have a `dataset.json` file into your `nnunet/raw/DatasetXXX` directory. 
	- See the [dataset.json section](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson) from the nnUNet to see a minimally working product.
	- The most important parts are the following:
		- `channel_names` properly set to your dataset especially if you have multi-modality data. You need to tell the nnUNet what is contrast `0`, `1`, etc.
		- `labels`: **MUST** have `"background": 0` as the first label (and with a lowercase b).
		- `numTraining`: must exact. This corresponds to the number of 'subjects'/datasets, not total number of files if you have multi-modality (e.g., 2 modalities with 50 subjects, `numTraining` == 50, not 100.). 
		- `file_ending`: specify if it is `.nii`, `.nii.gz`, etc. for your data.
		- The rest is not mandatory but can be helpful.
5) **See notes of step #8 if you don't want to do any modification to the normal pipeline** Run `nnUNetv2_extract_fingerprint -d XXX --verify_dataset_integrity`
6) Run `nnUNetv2_plan_experiment -d XXX -pl nnUNetPlannerResEncL -overwrite_plans_name YOUR_DESIRED_NAME`
	- The plan name you will specify here should be the same one used for the steps below.
	- If different preprocessings are to be tested/performed (e.g., wanna try the impact of different upsampling strategies), manually modify the "data_identifier" field inside the plan file produced for the corresponding nnUNet config you want to do this for to not erase the previously preprocessed data.
		- This is not a requirement for the first run, but don't forget to do it if you do a modification afterwards otherwise it will rewrite the preprocessed folder.
7) Manually modify your new plans file (in any IDE) to include the modifications you want.
	- I usually manually create a copy and change the name to keep a 'baseline' version of the plans and the new one with the modifications, but you don't have too.
8) Run `nnUNetv2_preprocess -d XXX -plans_name PLANS_NAME -c 3d_fullres -np 12`
	- **Note**: You can also simply run `nnUNetv2_plan_and_preprocess` command instead of steps #5 and 6 above to do all of them in one go. The main reason why we do them seprately is to modify the plans file manually at step #7.
		- Alternatively, as a sanity check if you do this single command for the 1st time, I recommend to run with `--no_pp` to detect any problems with the raw dataset.
	- `--verify_dataset_integrity` can be used only once for each raw dataset. If it works the 1st time, no need to reverify the dataset.
	- In order to save time, you can only choose one config for the `-c` flag (e.g., `-c 3d_fullres`) if you are not planning to do it anyways.
	- Could probably increase the number of CPU/cores (`-np 12`) allocated to increase processing time (ca. 1h30 for 414 3D images using 3d spline).
		- CPU/setup-dependent. We have 16 CPU cores so we can use many, but not every computer has that many.
9) Run the `script-nnunet-fold-trainings.py` script with its flags to script automatically the different folds, datasets, plans and/or even trainers. 
	- Quite powerful script that doesn't crash on errors and jump to the next iteration (doesn't influence the quality of the next iteration, will only require to rerun it at a later stage.)
	- **Important mention**: If you would happen to get data_loading issue similar to [this](https://github.com/MIC-DKFZ/nnUNet/issues/2712), type `nnUNet_compile=False` in your terminal and rerun the same exact command as you previously did. This could be an easy fix.
	- **Alternatively**, use the original nnUNet script  `nnUNetv2_train XXX 3d_fullres Y --npz -p nnUNetResEncUNetLPlans -tr nnUNetTrainerNoDA --val_best`
		- where `Y` is the fold number between 0 and 4 here.
		- **Note**: If performed with this command, you will have to manually start each fold. 
10) Once the training is completed (all folds and configurations), run `nnUNetv2_find_best_configuration XXX -c CONFIG-NAME -p PLANS-FILE -tr TRAINER-NAME `
	- The `-p` and `-tr` can be different based on the previous steps.
		- For `-p`, check in the `preprocessed` folder.
	- This step is to some extent irrelevant* if only one configuration was ran.
		- *However, this step will tell you exactly how to run the inference and postprocessing (it will print the command lines to run and save them in a .txt file in the results/DatasetXXX/ folder) which can be practical. 
11) Run the script `inf-postpro-eval-nnunet.py` with its corresponding flags to run (1) the inference, (2) the postprocessing on the inferred results and (3) evaluate these postprocessed results.
	- Check the documentation of this script to get more details since there are many possibilities (too many for a SOP). F. eks., the evaluation process can be skipped if only the inference and postprocessing are to be run.
	- The `.JSON` results file will be saved in the predictions folder provided above.
	- **Tip**: By using a different `dataset.json` file with less labels, you can compute the same metrics on a different number of labels if you want to exclude them from the analysis. Be sure to specify a different output file with `-o` to make sure you dont overwrite the previously calculated output file calculated on a different number of labels. 