from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class YLabelEncoder:
    def __init__(self, data_df, parameters_to_be_encode):
        self.data_df = data_df
        self.parameters_to_be_encode = parameters_to_be_encode

    def run_encoding(self):
        y_encoded_labels = []
        y_encoder = {}
        for p in self.parameters_to_be_encode:
            le = LabelEncoder()
            le = le.fit(list(self.data_df[p]))
            y_label = le.transform(list(self.data_df[p]))
            y_encoder[p + ' status'] = le
            if p == 'knee':
                y_label[np.where(y_label == 2)] = 0
            y_encoded_labels.append(y_label)

        y_encoded_labels = np.asarray(y_encoded_labels).T
        y_encoded_labels_df = pd.DataFrame(y_encoded_labels, columns=[j+' status' for j in self.parameters_to_be_encode])
        return y_encoder, y_encoded_labels_df

