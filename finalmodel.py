#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
#%%
df = pd.read_csv("data/train.csv")
X = df.drop(["id", "claim"], axis=1)
y = df["claim"]

#%%
X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                      y, 
                                                      train_size=0.7, 
                                                      test_size=0.3,
                                                      random_state=7,
                                                      stratify=y)
#%%
simplePreprocess = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=np.nan))
])

pcaPreprocess = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),
])

clusterPreprocess = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('cluster', FeatureAgglomeration(n_clusters=10))
])

featureExtraction = FeatureUnion(transformer_list=[
    ('simple', simplePreprocess),
    ('pca', pcaPreprocess),
    ('cluster', clusterPreprocess)
])

model = Pipeline(steps=[
    ('preprocess', featureExtraction),
    ('model', xgb.XGBClassifier(objective='binary:logistic', 
                                seed=7,
                                gamma=15.0,
                                learning_rate=0.2,
                                max_depth=8,
                                reg_lambda=15.0,
                                n_estimators=300,
                                use_label_encoder =False,
                                colsample_bytree=0.8
                                ))
])
#%%
featureExtraction.fit(X_train)

model.fit(X_train, 
          y_train, 
          model__verbose=True, 
          model__early_stopping_rounds=20,
          model__eval_metric='aucpr',
          model__eval_set=[(featureExtraction.transform(X_valid), y_valid)])
#%%
model.fit(X, 
          y, 
          model__verbose=True, 
          model__eval_metric='aucpr')

#%%
df = pd.read_csv("data/test.csv")
X = df.drop(["id"], axis=1)
#%%
y_pred = model.predict_proba(X)[:,1]
submission = pd.DataFrame({'id': df['id'], 'claim': y_pred})
#%%
submission.to_csv("data/submission.csv", index=False)

