import pandas as pd
import numpy as np
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

import umap
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

class MBPhenixScaler(BaseEstimator, TransformerMixin):
  def __init__(self, ignored_features: List[str] = [], target_weight = 0.9):
    self.columns = None
    self.ignored_features = ignored_features
    self.scaler = umap.UMAP(n_components=10, verbose=False,metric='cosine',n_epochs=1000,min_dist=0.1,n_neighbors=200,
                                random_state=123, target_weight=target_weight)
    self.new_matrix = None

  def fit(self, X, y=None):
      """
      Fits the scaler to the data.

      Parameters:
        X (pandas.DataFrame): The input data to fit the scaler on.
        y (pandas.Series or None): The target variable. Default is None. Strongly recommend supervised learning.

      Returns:
        self (Scaler): The fitted scaler object.
      """
      relevant_columns = X.loc[:, ~X.columns.isin(self.ignored_features)]
      self.columns = [c for c in relevant_columns.columns]
      self._fit_not_ignored(relevant_columns, y)
      
      return self

  def _fit_not_ignored(self, X, y=None, t=3, decay=1, metric='euclidean', knn=10):
    umap_data = self.scaler.fit_transform(X.values, y)

    distance_matrix =pdist(umap_data, metric)
    distance_matrix = (squareform(distance_matrix))
    D = distance_matrix
    n,m = D.shape
    E = np.zeros((m,m))
    
    knn_dst = np.sort(distance_matrix, axis=1)

    epsilon = knn_dst[:,knn]
    pdx_scale = (distance_matrix / epsilon).T
    
    E = np.exp(-1 * ( pdx_scale ** decay))
    A = (E + E.T)
    
    diff_deg = np.diag(np.sum(A,0))
    diff_op = np.dot(np.diag(np.diag(diff_deg)**(-1)),A)

    self.new_matrix =  np.linalg.matrix_power(diff_op, t)

  def transform(self, X) -> pd.DataFrame:
    """Transforms the features.

    Arguments:
      X (pd.DataFrame): The data to be transformed.

    Returns:
      pd.DataFrame: The transformed data.
    """
    missing_columns = [c for c in self.columns if c not in list(X.columns)]

    if len(missing_columns) / len(self.columns) > 0.2:
        print(f'More than 20 percent of columns not found in  {self.__class__}. {len(X.columns)} available columns: {list(X.columns)}.')

    X = X.copy()

    for m in missing_columns:
      X[m] = 0.0

    relevant_columns = X.loc[:, ~X.columns.isin(self.ignored_features)].copy()
    relevant_columns = relevant_columns[self.columns]

    return self._transform_not_ignored(relevant_columns).join(X.loc[:, X.columns.isin(self.ignored_features)])

  def _transform_not_ignored(self, X) -> pd.DataFrame:
    """Uses mb-PHENIX to transfrom features. (DOI: 10.1093/bioinformatics/btad706)

    Arguments:
      X (pd.DataFrame): The data to be transformed.

    Returns:
      pd.DataFrame: The transformed data.
    """

    data_new = np.array(np.dot(self.new_matrix, X))
            
    Matix_col_genes_row_cell2 = (X +1) - X
    Matix_col_genes_row_cell2 = Matix_col_genes_row_cell2 - Matix_col_genes_row_cell2
    Matix_impu = Matix_col_genes_row_cell2 + data_new

    sc_PHENIX = Matix_impu 
    
    return (sc_PHENIX)


  def get_support(self) -> List[str]:
    """Returns a list of selected features

    Returns:
        List[str]: list of selected features
    """
    return self.columns
