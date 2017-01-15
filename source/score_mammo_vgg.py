from __future__ import print_function

import pandas as pd
import keras.models as km

import utils.constants as c
import utils.simple_loader as sl

if __name__ == '__main__':
    # Load up the trained model and evaluate it.
    model = km.load_model(c.MODELSTATE_DIR + '/' + c.MODEL_FILENAME)
    model.load_weights(c.MODELSTATE_DIR + '/' + c.WEIGHTS_FILENAME)

    sl = sl.SimpleLoader()
    predicted_vals = model.predict(sl.imgs)

    images_fields = ['subjectId', 'laterality']
    images_meta = pd.read_csv(c.IMAGES_CROSSWALK_FILENAME, sep="\t", na_values='.', usecols=images_fields)

    # Need to temporarily keep multiple confidence scores for each subjectId and laterality as there
    # could be multiple mammograms per breast for each subject.
    results = pd.DataFrame(columns=['subjectId', 'laterality', 'confidence_sum', 'num_scores'])
    results.subjectId = results.subjectId.astype(int)
    results.laterality = results.laterality.astype(str)
    results.num_scores = results.num_scores.astype(int)

    curr_index = 0
    for index, row in images_meta.iterrows():
        new_prediction = predicted_vals[curr_index][0]

        result_row = results[(results.subjectId == row['subjectId']) &
                             (results.laterality == row['laterality'])]
        if result_row.empty:
            # Found new subject and laterality. Add a new row to the DataFrame
            new_row = pd.DataFrame([[row['subjectId'], row['laterality'], new_prediction, 1]],
                                   columns=['subjectId', 'laterality', 'confidence_sum', 'num_scores'])
            results = results.append(new_row, ignore_index=True)
        else:
            # Add a new prediction to existing subject and laterality.
            results.loc[(results.subjectId == row['subjectId']) &
                        (results.laterality == row['laterality']), 'confidence_sum'] += new_prediction
            results.loc[(results.subjectId == row['subjectId']) &
                        (results.laterality == row['laterality']), 'num_scores'] += 1

        curr_index += 1

    # TODO:
    # Instead of just averaging the scores for each subject and laterality we could take
    # the max predicted score so we output 1 when at least one mammogram shows potential to show
    # breast cancer.
    results['confidence'] = results['confidence_sum'] / results['num_scores']
    print(results)

    # Ready to output the relevant columns to tsv file now.
    output_fields = ['subjectId', 'laterality', 'confidence']
    results.to_csv(c.OUTPUT_FILE, sep='\t', columns=output_fields, index=False)
