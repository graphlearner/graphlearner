from sklearn.base import BaseEstimator, TransformerMixin


class ColumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features=None):
        if features is None:
            features = []
        self.features = features

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        data = X if (self.features is None) else X[self.features].copy()
        # print(f'Fitting with columns: {data.columns}')

        return data

    def transform(self, X, y=None):
        data = X if (self.features is None) else X[self.features].copy()

        return data
