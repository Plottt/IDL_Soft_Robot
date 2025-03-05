# IDL_Soft_Robot

Project to do object classification using a soft robot finger for 11-785 Intro to IDL.

spiral_finger_data_reader.py Computer-side code to read from serial port and write to .txt file.

IDL_Data_Collection.ino Microcontroller (Arduino) code to run data collection sequence and print to Serial.

## environment preparation [plain]
Only need to download pandas, scikit-learn, and torch. You can DIY on your own env, or use the following commands: 
```
conda create -n IDL python=3.8
pip install -r requirements.txt
```
## Training code
- `dataloader` in `utils/dataloader.py`
    - read panda dataframes from `*.txt` files
    - organize data according to columes
    - [TODO] extract features from data
        - currently only return `means` for each column
- train/eval loop in `train.ipynb`
    - implement basic MLP for classification based on contact **features**
    - implement vanilla train loop
    - use `scilearn.classification_report` to report metrics
        - currently achieve 0.0 validation accuracy :( 