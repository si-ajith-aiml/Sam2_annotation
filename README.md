# Sam2_annotation
Implemented annotation tool and predicted object by sam2 model gives object centroid in csv  and visual displayed videos 

# 1. go to directory
cd checkpoints

# 2. download weights file by
download_weights.py 

# 3. Add videos in input videos folder 

# 4. run pre_annotator_tool.py
It will create & save all data in preprocessed_data folder .

# 5. run sam2_predictor.py
It will create & save all data in Sam2_predictor_data folder .
Inside 
for each video if (x , y) coordinates is slected then it will save in Object_present_files folder.
for each video if (x , y) coordinates is not slected then it will save in No_object_present_files folder.
for each video ifany error occurred then it will save in error_files folder.

# to save Annotated Videos
should be always (save_annotate = 1 )in sam2_predictor.py inside
it will save videos in  Sam2_predictor_data/Sam2_visualization_annoted_clips

