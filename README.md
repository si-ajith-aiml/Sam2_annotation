# Sam2 Annotation

An annotation tool that utilizes the SAM2 model to predict objects and generate their centroid coordinates in a CSV file. It also visualizes the annotated videos.

---

## **Requirements**
- **Python**: 3.13.0  
- **CUDA**: Minimum **11.8** or higher  
- **Dependencies**: `torch`, `torchaudio`, `torchvision`

---

## **Installation Steps**

### **1. Create a Conda Environment**
```bash
conda create --name sam2 python=3.10 -y
conda activate sam2


////steps to follow 

--Create conda env  copy
conda create --name sam2 python=3.10 -y
then
conda activate sam2

--for CUDA 11.8 copy 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
then
pip install -r requirements.txt

1. go to directory
cd checkpoints

2. download weights file
download_weights.py  

3. Add videos in input videos folder 

4. run pre_annotator_tool.py
It will create & save all data in preprocessed_data folder .

5. run sam2_predictor.py
It will create & save all data in Sam2_predictor_data folder .
Inside 
for each video if (x , y) coordinates is slected then it will save in Object_present_files folder.
for each video if (x , y) coordinates is not slected then it will save in No_object_present_files folder.
for each video ifany error occurred then it will save in error_files folder.

6.to save Annotated Videos
should be always (save_annotate = 1 )in sam2_predictor.py inside
it will save videos in  Sam2_predictor_data/Sam2_visualization_annoted_clips

