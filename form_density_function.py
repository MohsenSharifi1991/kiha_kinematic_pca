import pandas as pd
import numpy as np
import pickle as pk
from sklearn.preprocessing import StandardScaler
scaler_save = True
activity_densities_save = True
pcs = pd.read_csv('./result/all_pca_patient_variables_singlegait.csv')
pcs_std = pcs.copy()
activities = ['Gait', 'Stair Ascent', 'Stair Descent', 'STS']
activity_densities = {'Gait':{}, 'Stair Ascent':{}, 'Stair Descent':{}, 'STS':{}}
for a, activity in enumerate(activities):
    # calculated density function of PCs per activity
    pcs_columns = [c for c in pcs.columns if 'Gait' in c]
    scaler = StandardScaler().fit(pcs[pcs_columns])
    pcs_std[pcs_columns] = scaler.fit_transform(pcs[pcs_columns])
    oa_mu = np.mean(pcs_std[pcs_columns][pcs_std['knee'] == 'OA']).values
    oa_cov = np.cov(pcs_std[pcs_columns][pcs_std['knee'] == 'OA'].T, bias=True)
    tka_mu = np.mean(pcs_std[pcs_columns][pcs_std['knee'] == 'BiTKA']).values
    tka_cov = np.cov(pcs_std[pcs_columns][pcs_std['knee'] == 'BiTKA'].T, bias=True)
    if scaler_save:
        pk.dump(scaler, open('./cache/pcs_scaler_' + activity + '.pkl', 'wb'))
    activity_densities[activity]['oa_mu'] = oa_mu
    activity_densities[activity]['oa_cov'] = oa_cov
    activity_densities[activity]['tka_mu'] = oa_mu
    activity_densities[activity]['tka_cov'] = tka_cov

if activity_densities_save:
    pk.dump(activity_densities, open('./cache/activity_densities.pkl', 'wb'))

# later reload the pickle file
pca_reload = pk.load(open("pca.pkl",'rb'))
result_new = pca_reload.inverse_transform(X)


import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print(clf_pf.predict([[-0.8, -1]]))