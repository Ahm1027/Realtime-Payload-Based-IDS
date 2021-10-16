#!/usr/bin/env python
import os
import re
import unicodedata
import time
import json
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from flask import Flask
from flask_apscheduler import APScheduler

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import joblib

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch_dsl import Search


#=================GLOBAL VARS====================
app = Flask(__name__)
port = 9600

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()
SEARCH_AND_PREDICT = 'search-and-predict'

#=================MODEL FUNCTIONS====================

def vectorize_payload(payload):
    vec_255 = [0]*255
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1))
    vectorized = vectorizer.fit_transform([payload])
    mapped = list(zip(vectorizer.get_feature_names(), vectorized.sum(0).getA1()))
    for x in mapped:
    	if (ord(x[0]) < 256):
    		vec_255[ord(x[0])] = x[1]
    return vec_255

def calc_frequency(payloads_vectors):
    features_total = 0
    for feature in payloads_vectors:
        features_total += feature
    features_total *= 1.0
    payloads_vectors = [(x/features_total) for x in payloads_vectors]
    return np.array(payloads_vectors, dtype=np.float16)

def calc_mean(frequency_vectors):
    return np.mean(frequency_vectors, axis=0)

def calc_sd(features_frequency):
    return np.std(features_frequency, axis=0)

def calc_cov_mat(features_frequency):
    mean_shift = np.subtract(features_frequency, np.mean(features_frequency)) 
    return np.matmul(mean_shift, mean_shift.T), mean_shift

def cal_pca(features_data):
    pca = PCA(20)
    pca.fit_transform(features_data)
    return pca

def format_covs(compressed_data, compressed_data_mean):
    covs = []
    for x in compressed_data:
    	cov = cal_covariance(x, compressed_data_mean, feature_population)
    	covs.append(cov)
    return np.array(covs)

def cal_covariance(feature, mean):
	mean_shift = feature - mean
	cov_mat = (mean_shift*mean_shift.T)
	return cov_mat

def calc_mahalanobis_dist(feature_x1, feature_x2,var_a,var_b):
    feature_diff = feature_x1-feature_x2
    feature_diff = feature_diff * feature_diff
    if(var_a+var_b==0):
    	return 0;
    return feature_diff/(var_a+var_b)

def calc_cov(feature, total_features=20):
	inter = (feature - feature/total_features)
	return inter*inter

def var_of_features(xi,mean):
	return (xi-mean)**2;


def construct_mdm(features, features_mean):
	mdm = []
	i=0
	for i in range(len(features)):
		feature_map = []
		for j in range(len(features)):
			var_a=var_of_features(features[i],features_mean[i])
			var_b=var_of_features(features[j],features_mean[j])

			feature_map.append(calc_mahalanobis_dist(features[i], features[j],var_a,var_b))
		mdm.append(feature_map)
	return np.array(mdm)

def cal_weight(real_payload_map, payloads_map_mean, payloads_map_variances):
	return (np.square(real_payload_map - payloads_map_mean)/payloads_map_variances)

#=================SCHEDULER MANAGE FUNCTIONS====================

def pauseAPScheduler():
    scheduler.pause_job(id=SEARCH_AND_PREDICT)
    return 'Scheduler Paused', 200


def resumeAPScheduler():
    scheduler.resume_job(id=SEARCH_AND_PREDICT)
    return 'Scheduler Paused', 200


#=================MAIN MODEL FUNCTION====================

def search_and_predict():
	pauseAPScheduler()
	# df = pd.read_csv(r'SQLi.csv', sep=',', error_bad_lines=False)
	df = pd.read_json('sqlinjections.json', lines=True)

	if len(df) >= 2:

		# df_reduced = df[df['Label'] == '1']
		# df_reduced = df_reduced[df_reduced['Sentence'].notnull()].copy()
		# df_reduced['Sentence'] = df_reduced['Sentence'].apply(lambda x: str(x))
		# df_reduced.drop(columns=['Unnamed: 2', 'Unnamed: 3'], inplace=True)
		# print(df_reduced.head())
		# number_of_observations = len(df_reduced)

		# Vectorizing and calculating frequencies for payloads
		# vectorized = np.vstack(df_reduced['Sentence'].apply(lambda x: vectorize_payload(x)))
		# features_frequency = np.vstack([calc_frequency(x) for x in vectorized])

		x_train, x_test = np.split(df, [int(.75*len(df))])

		x_train['payload'] = x_train['payload'].apply(lambda x: str(x))
		print(x_train.head())
		number_of_observations = len(x_train)

		# Vectorizing and calculating frequencies for payloads
		vectorized = np.vstack(x_train['payload'].apply(lambda x: vectorize_payload(x)))
		features_frequency = np.vstack([calc_frequency(x) for x in vectorized])

		#Calculating meanshift
		# features_mean = calc_mean(vectorized)
		# mean_shift = vectorized - features_mean
		# print(mean_shift.shape)

		#Covariance Matrix for mean_shift for calculating PCA
		# cov_mat = np.matmul(mean_shift, mean_shift.T)/(number_of_observations-1)
		# print(cov_mat.shape)
		# print(cov_mat)

		#Using Singular Value Decomposition TruncatedSVD for calculated PCA = 8
		# compressed_SVD = TruncatedSVD(8)
		# compressed_data = compressed_SVD.fit_transform(mean_shift)
		# print("==========Compressed Data==========")
		# print("Compressed Data Shape: ",compressed_data.shape)
		# print("Compressed Data: ",compressed_data)
		# print("Compressed explained_variance_ratio: ",compressed_SVD.explained_variance_ratio_)
		# print("Compressed singular_values: ",compressed_SVD.singular_values_)
		# print("Compressed explained_variance_ratio: ",compressed_SVD.explained_variance_ratio_.sum())
		# print("Cumsum: ",np.cumsum(compressed_SVD.explained_variance_ratio_))

		compressed_PCA = PCA(n_components=20)
		compressed_data = compressed_PCA.fit_transform(features_frequency)
		print("==========Compressed Data==========")
		print("Compressed Data Shape: ",compressed_data.shape)
		print("Compressed Data: ",compressed_data)
		print("Compressed explained_variance_ratio: ",compressed_PCA.explained_variance_ratio_)
		print("Compressed singular_values: ",compressed_PCA.singular_values_)
		print("Compressed explained_variance_ratio: ",compressed_PCA.explained_variance_ratio_.sum())
		print("Cumsum: ",np.cumsum(compressed_PCA.explained_variance_ratio_))
		joblib_file = "compressed_data.pkl"
		joblib.dump(compressed_PCA, joblib_file)

		#Calculating meanshift for reduced features
		features_compressed_mean = calc_mean(compressed_data)
		mean_shift_compressed = np.transpose(compressed_data - features_compressed_mean)
		print(mean_shift_compressed.shape)

		#Covariance Matrix for mean_shift_reduced
		cov_mat_reduced = np.matmul(mean_shift_compressed, mean_shift_compressed.T)/(number_of_observations-1)
		print(cov_mat_reduced.shape)
		print(cov_mat_reduced)

		#Calculating mean of compressed data for covariances
		# compressed_data_mean = calc_mean(compressed_data)
		# covariances = format_covs(compressed_data, compressed_data_mean)

		# Constructing payload map using covariance matrix for reduced data
		payloads_map = []
		for x in compressed_data:
			payloads_map.append(construct_mdm(x, features_compressed_mean))
		payloads_map = np.array(payloads_map)
		# print("Payloads Map")
		# print("Payload map shape: ",payloads_map.shape)

		# print(payloads_map)
		payloads_map_mean = calc_mean(payloads_map)
		payloads_map_vars = np.var(payloads_map, axis=0)
		payloads_map_sd = calc_sd(payloads_map)
		# print("Payloads Map Mean: ",payloads_map_mean)
		# print("Payloads Map Mean: ",payloads_map_vars)
		# print("Payloads Map Mean: ",payloads_map_sd)
		np.save("payloads_map_mean.npy", payloads_map_mean)
		np.save("payloads_variances_mean.npy", payloads_map_vars)
		np.save('payloads_map_sd.npy', payloads_map_sd)
		
	else:
		resumeAPScheduler()


#=================SCHEDULER CONF====================

scheduler.add_job(id=SEARCH_AND_PREDICT, func=search_and_predict,
                  trigger='interval', seconds=1)

@app.route('/status')
def testServer():
    return 'Status 200'


if __name__ == '__main__':
    app.run(debug=True, port=port, use_reloader=False)