from sklearn.linear_model import LinearRegression
from Bgolearn import BGOsampling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import pandas as pd
import cmu
from typing import Optional


class Mobo(object):
    """
    Multi Objective Bayesian Optimization.

    Parameters
    ----------
    mission

    Attributes
    ----------

    Examples
    --------
    """
    def __init__(self):
        pass

    def fit(self, X, y, visual_data, mission: Optional[str]=None, method='HV', number=1, objective: Optional[str]=None, ref_point=Optional[list]=None):
        Xtrain = X
        Ytrain = y
        Xtest=visual_data 
        target_names = Ytrain.columns.tolist()

        if objective == 'max':
            pareto_front = find_non_dominated_solutions(Xtrain, target_names)
            pareto_front = pd.DataFrame(pareto_front, columns=target_names)

            kernel = RBF(length_scale=1.0)
            gp_model = GaussianProcessRegressor(kernel=kernel)
            gp_model.fit(Xtrain, Ytrain)
            Ypred, Ystd = gp_model.predict(Xtest, return_std=True)
            Ypred = pd.DataFrame(Ypred, columns=Ytrain.columns.tolist())

            if method == 'HV':
                    HV_values = []
                    for i in range(Ypred.shape[0]):
                        i_Ypred = Ypred.iloc[i]
                        Ytrain_i_Ypred = Ytrain.append(i_Ypred)
                        i_pareto_front = find_non_dominated_solutions(Ytrain_i_Ypred.values, Ytrain_i_Ypred.columns.tolist())
                        i_HV_value = dominated_hypervolume(i_pareto_front, ref_point)
                        HV_values.append(i_HV_value)
                    
                    HV_values = pd.DataFrame(HV_values, columns=['HV values'])
                    HV_values.set_index(Xtest.index, inplace=True)

                    max_idx = HV_values.nlargest(number, 'HV values').index
                    recommend_point = Xtest.loc[max_idx]
                    Xtest = Xtest.drop(max_idx)
                    print('The maximum value of HV:', HV_values.loc[max_idx]) 
                    print('The recommended point is :', recommend_point)
            elif method == 'EHVI':
                pass
            
    def preprocess(data, target_number, normalize: Optional[str]=None):
        df = pd.read_csv(data)
        X = df.iloc[:,:-target_number].values
        y = df.iloc[:,-target_number:]
        if normalize == 'StandardScaler':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif normalize == 'MinMaxScaler':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        else:
            X = X        

        return X, y