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
payloads_map_mean = np.load('payloads_map_mean.npy')
payloads_map_vars = np.load('payloads_variances_mean.npy')
payloads_map_sd = np.load('payloads_map_sd.npy')
compressed_PCA = joblib.load('compressed_data.pkl')

#=================MODEL FUNCTIONS====================

def vectorize_payload(payload):
    vec_255 = [0]*255
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1))
    vectorized = vectorizer.fit_transform([payload])
    mapped = list(zip(vectorizer.get_feature_names(), vectorized.sum(0).getA1()))
    for x in mapped:
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
[3, 4, 6, 3, 6, 8, 5, 2]
def cal_pca(features_data):
    pca = PCA(10)
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

def calc_mahalanobis_dist(feature_x1, feature_x2, cov_x1, cov_x2):
    feature_diff = feature_x1-feature_x2
    feature_diff = feature_diff*feature_diff
    if cov_x1+cov_x2 == 0:
        return 0
    return feature_diff/(cov_x1+cov_x2)

def construct_mdm(features, covariances):
    mdm = []
    i=0
    for i, feature1 in enumerate(features):
        feature_map = []
        for j, feature2 in enumerate(features):
        	feature_map.append(calc_mahalanobis_dist(feature1, feature2, covariances[i][i], covariances[j][j]))
        mdm.append(feature_map)
    return np.array(mdm)

def cal_weight(real_payload_map, payloads_map_mean, payloads_map_variances):
	substracted = np.square(real_payload_map - payloads_map_mean)
	return np.divide(substracted, payloads_map_variances, out=np.zeros_like(substracted), where=payloads_map_variances!=0)

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
	df = pd.read_json(r'2021-10-08.json', convert_dates=True, lines=True)
	if len(df) >= 2:

		df.dropna(inplace=True)
		number_of_observations = len(df)
		# Vectorizing and calculating frequencies for payloads
		vectorized = np.vstack(df['request'].apply(lambda x: vectorize_payload(x)))
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

		compressed_data = compressed_PCA.transform(features_frequency)
		print("==========Compressed Data==========")
		print("Compressed Data Shape: ",compressed_data.shape)
		print("Compressed Data: ",compressed_data)
		print("Compressed explained_variance_ratio: ",compressed_PCA.explained_variance_ratio_)
		print("Compressed singular_values: ",compressed_PCA.singular_values_)
		print("Compressed explained_variance_ratio: ",compressed_PCA.explained_variance_ratio_.sum())
		print("Cumsum: ",np.cumsum(compressed_PCA.explained_variance_ratio_))

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
		real_payload_maps = []
		for x in compressed_data:
			real_payload_maps.append(construct_mdm(x, cov_mat_reduced))
		real_payload_maps = np.array(real_payload_maps)
		print("Payloads Map")
		print("Payload map shape: ",real_payload_maps.shape)
		payloads_sd_sum = np.sum(payloads_map_sd)
		positive_threshold = 3*payloads_sd_sum
		negative_threhold = -3*payloads_sd_sum
		print("Payloads Mean Map: ",payloads_map_mean)
		print("Payloads Map SD: ",payloads_map_sd)

		mean_sum = np.sum(payloads_map_mean)
		positive_threshold += mean_sum
		negative_threhold += mean_sum
		print("positive_threshold: ",positive_threshold)
		print("negative_threhold: ",negative_threhold)

		for x in real_payload_maps:
			try:

				weight = cal_weight(x, payloads_map_mean, payloads_map_vars)
				print("Weight Matrix: ",weight)
				print("Payload Weight: ",np.sum(weight))
				print("Payload Already Exists!")
				# sum =0
				# for x in weight:
				# 	x = x[~pd.isnull(x)]
				# 	sum += np.sum(x)
				# print("Payload Weightage: ",sum)
				print("Fetching next payload for analysis...")
				time.sleep(5)
			except:
				continue

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
