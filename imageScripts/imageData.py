import os
import json

likelihood_scores = {'UNKNOWN': '0', 'VERY_UNLIKELY': 0, 'UNLIKELY': 1, 'POSSIBLE': 2, 'LIKELY': 3, 'VERY_LIKELY': 4}

def read_json(path):
    json_files = [pos_json for pos_json in os.listdir(path)]
    print("Total Number of JSON Files: ", len(json_files))
    return json_files


def construct_df_from_json(json_files, df, path_to_json):
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)

            full_id = js.partition('.json')[0].partition('-')
            pet_id = full_id[0]
            photo_num = full_id[2]

            faceAnnotations = json_text.get('faceAnnotations', 'N/A')
            detectionConfidence = 0
            joyLikelihood = 0
            sorrowLikelihood = 0
            angerLikelihood = 0
            surpriseLikelihood = 0
            underExposedLikelihood = 0
            blurredLikelihood = 0
            headwearLikelihood = 0

            if faceAnnotations != 'N/A':
                detectionConfidence = faceAnnotations[0]['detectionConfidence']
                joyLikelihood = likelihood_scores[faceAnnotations[0]['joyLikelihood']]
                sorrowLikelihood = likelihood_scores[faceAnnotations[0]['sorrowLikelihood']]
                angerLikelihood = likelihood_scores[faceAnnotations[0]['angerLikelihood']]
                surpriseLikelihood = likelihood_scores[faceAnnotations[0]['surpriseLikelihood']]
                underExposedLikelihood = likelihood_scores[faceAnnotations[0]['underExposedLikelihood']]
                blurredLikelihood = likelihood_scores[faceAnnotations[0]['blurredLikelihood']]
                headwearLikelihood = likelihood_scores[faceAnnotations[0]['headwearLikelihood']]

            labels = 0
            dog_like_mammal = 0
            kitten = 0
            puppy = 0
            cat_like_mammal = 0
            european_shorthair = 0
            fauna = 0
            carnivoran = 0
            cat = 0
            snout = 0
            domestic_short_haired_cat = 0
            dog = 0
            dog_breed_group = 0
            small_to_medium_sized_cats = 0
            aegean_cat = 0
            sporting_group = 0
            street_dog = 0
            whiskers = 0
            dog_breed = 0

            labelAnnotations = json_text.get('labelAnnotations', 'N/A')
            if labelAnnotations != 'N/A':
                for label in labelAnnotations:
                    labels += 1
                    if label['description'] == 'dog like mammal':
                        dog_like_mammal = label['score']
                    elif label['description'] == 'kitten':
                        kitten = label['score']
                    elif label['description'] == 'puppy':
                        puppy = label['score']
                    elif label['description'] == 'cat like mammal':
                        cat_like_mammal = label['score']
                    elif label['description'] == 'european shorthair':
                        european_shorthair = label['score']
                    elif label['description'] == 'fauna':
                        fauna = label['score']
                    elif label['description'] == 'carnivoran':
                        carnivoran = label['score']
                    elif label['description'] == 'cat':
                        cat = label['score']
                    elif label['description'] == 'snout':
                        snout = label['score']
                    elif label['description'] == 'domestic short haired cat':
                        domestic_short_haired_cat = label['score']
                    elif label['description'] == 'dog':
                        dog = label['score']
                    elif label['description'] == 'dog breed group':
                        dog_breed_group = label['score']
                    elif label['description'] == 'small to medium sized cats':
                        small_to_medium_sized_cats = label['score']
                    elif label['description'] == 'aegean cat':
                        aegean_cat = label['score']
                    elif label['description'] == 'sporting group':
                        sporting_group = label['score']
                    elif label['description'] == 'street dog':
                        street_dog = label['score']
                    elif label['description'] == 'whiskers':
                        whiskers = label['score']
                    elif label['description'] == 'dog breed':
                        dog_breed = label['score']

            df.loc[index] = [pet_id, photo_num, detectionConfidence, joyLikelihood,
                             sorrowLikelihood,
                             angerLikelihood, surpriseLikelihood,
                             underExposedLikelihood,
                             blurredLikelihood, headwearLikelihood,
                             labels, street_dog,
                             dog_like_mammal, aegean_cat, carnivoran,
                             small_to_medium_sized_cats,
                             cat_like_mammal, dog_breed, snout,
                             whiskers, domestic_short_haired_cat,
                             puppy, dog_breed_group,
                             fauna, kitten, european_shorthair, dog, sporting_group, cat]

    return df


def read_most_common_labels(json_files, cutoff, path_to_json):
    all_labels = []
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
            labelAnnotations = json_text.get('labelAnnotations', 'N/A')
            if labelAnnotations != 'N/A':
                for label in labelAnnotations:
                    all_labels.append(label['description'])

    unique_labels = set(all_labels)

    important_words = {}
    for l in unique_labels:
        count = all_labels.count(l)
        word = [l]
        if count in important_words:
            important_words[count].append(l)
        else:
            important_words[count] = word

    # sorted_d = sorted(important_words.items(), key=operator.itemgetter(0), reverse=True)

    filtered_words = {k: v for k, v in important_words.items() if k > cutoff}
    return filtered_words
