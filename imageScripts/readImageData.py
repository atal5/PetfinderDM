import pandas as pd
from src.main.python.imageData import *


print(likelihood_scores)

# # Testing one file
# with open('/Users/Rima/Documents/Q2/petfinderkaggle/petfinder-adoption-prediction/train/train_metadata/8b01f5e06-2.json') as json_data:
#     data = json.load(json_data)
#
# print(data)

path_to_json = '/Users/Rima/Documents/Q2/petfinderkaggle/petfinder-adoption-prediction/train/train_metadata/'
files = read_json(path_to_json)
filtered_words = read_most_common_labels(files, 8000, path_to_json)

label_columns = [v for k, [v] in filtered_words.items()]
df_columns = ['petid', 'photo_num', 'detectionConfidence', 'joyLikelihood', 'sorrowLikelihood', 'angerLikelihood',
              'surpriseLikelihood', 'underExposedLikelihood',
              'blurredLikelihood', 'headwearLikelihood', 'labels'] + label_columns

print(df_columns)

# manually setting this from result above - to make sure order is same in which data is entered in DF
df_columns = ['petid', 'photo_num', 'detectionConfidence', 'joyLikelihood',
              'sorrowLikelihood', 'angerLikelihood', 'surpriseLikelihood',
              'underExposedLikelihood', 'blurredLikelihood', 'headwearLikelihood',
              'labels', 'street dog', 'dog like mammal', 'aegean cat', 'carnivoran',
              'small to medium sized cats', 'cat like mammal', 'dog breed', 'snout',
              'whiskers', 'domestic short haired cat', 'puppy', 'dog breed group',
              'fauna', 'kitten', 'european shorthair', 'dog', 'sporting group', 'cat']

pet_image_data = pd.DataFrame(columns=df_columns)
pet_image_data = construct_df_from_json(files, pet_image_data, path_to_json)
pet_image_data.to_pickle("../../../data/pet_image_data.pkl")
