## Goal

conducting PCA analysis on kinematic data to identify the relationship between PCs and patient demographic data 
 
## Research question?

1) can ML model predict the pelvic kinematic accurate (previous study)?
2) does training on both synthetic and measured data improve kinematic prediction?
3) Does training on orientation augmented data improve generalizability of the model relative to IMU orientation?
4) Which model BiLSTM vs Transformer Encoder?
5) Can we develop a pipeline to generate targeted synthetic data (model generalization)?

## Method
#### a) preprocessing
0) activities:
-   Gait
-   Stair Ascent
-   Stair Descent
-   sit 2 stand
-   sit max flexion
-   sit hip flexion-extension
-   lunge?
1) update dataset
- export trc file form text file
- run ik baseline
- run ik segmentation

2) ensure the marker location on pelvic before run IK

#### b) data exploration 
goal: how data set distribution are varied and how we can use them for target augmentation or domain generalization 
1) dimension reduction: PCA, tSNE, umap, spectral, ... 
1) clustering kinematic and imu 
- activity based decomposition using LDA, KL, JS Diveregnace, mixture of gaussian 
- low and high risk based decomposition:

#### c) augmentation


#### d) ml
1) generate synthetic data based on previous method: non-label 
2) generate IMU orientation augmentation: label preserving 
3) train model 

### Evaluation

### References
[1] https://github.com/amber0309/Domain-generalization#Survey-papers 
[2] https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html


["STS", "Sit Max Flexsion", "Hip Fle-Ext-R", "Hip Fle-Ext-L", "lunge-R Forward", "lunge-L Forward"]
  "selected_opensim_labels": ["pelvis_tilt", "L1_L2_IVD_bendjnt_r1", "L2_L3_IVD_bendjnt_r1",
  "L3_L4_IVD_bendjnt_r1", "L4_L5_IVD_bendjnt_r1", "flex_extension"],
  
  
 
git init
git status
git commit -m "my commit"
git add .
git remote add origin https://github.com/MohsenSharifi1991/opensim_spinopelvic_kinematic.git

linux: virtualenv --python python3.7 env
git push origin main 
git pull origin main

  "xsensimu_sensor_list_all": ["Hand Left", "Right Hand","PELVIS",
                              "Right uLEG", "Right lLeg","Right Foot",
                              "Left uLEG", "Left lLeg", "Left Foot"],
  "osimimu_sensor_list_all" : ["C7IMU", "T12IMU","PelvisIMU",
                              "RUpLegIMU", "RLoLegIMU","RFootIMU",
                              "LUpLegIMU","LLoLegIMU","LFootIMU"],
                              
  "xsensimu_sensor_list": ["Hand Left", "Right Hand","PELVIS", "Right uLEG", "Right lLeg","Right Foot",
                              "Left uLEG", "Left lLeg", "Left Foot"],
  "osimimu_sensor_list" : ["C7IMU","T12IMU","PelvisIMU",
                              "RUpLegIMU", "RLoLegIMU","RFootIMU",
                              "LUpLegIMU","LLoLegIMU","LFootIMU"],
  "imu_sensor_list": ["C7","T12", "P", "RT", "RS", "RF", "LT", "LS", "LF"],
  
    "train_subjects": ["S09","S10", "S11", "S12", "S13", "S15", "S16", "S17","S18", "S19","S20",
                     "S21","S22", "S23", "S25", "S26", "S27", "S28", "S29",
                      "S30", "S31", "S32", "S33", "S34","S35", "S36", "S37", "S38"],
  "test_subjects": ["S39"],
  ["Gait", "Stair Up-Down-R","Stair Up-Down-L","STS"],