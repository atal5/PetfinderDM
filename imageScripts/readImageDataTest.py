from src.main.python.imageData import *
import pandas as pd

path_to_json = '/Users/Rima/Documents/Q2/petfinderkaggle/petfinder-adoption-prediction/test/test_metadata/'
test_files = read_json(path_to_json)
df_columns = ['petid', 'photo_num', 'detectionConfidence', 'joyLikelihood',
              'sorrowLikelihood', 'angerLikelihood', 'surpriseLikelihood',
              'underExposedLikelihood', 'blurredLikelihood', 'headwearLikelihood',
              'labels', 'street dog', 'dog like mammal', 'aegean cat', 'carnivoran',
              'small to medium sized cats', 'cat like mammal', 'dog breed', 'snout',
              'whiskers', 'domestic short haired cat', 'puppy', 'dog breed group',
              'fauna', 'kitten', 'european shorthair', 'dog', 'sporting group', 'cat']
pet_image_data_test = pd.DataFrame(columns=df_columns)
pet_image_data_test = construct_df_from_json(test_files, pet_image_data_test, path_to_json)
pet_image_data_test.to_pickle("../../../data/pet_image_data_test.pkl")