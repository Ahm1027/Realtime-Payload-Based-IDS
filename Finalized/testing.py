#!/usr/bin/env python
import os
import re
import unicodedata
import time
import json
import random
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    	if ord(x[0]) < 255:
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

def cal_pca(features_data):
    pca = PCA(10)
    pca.fit_transform(features_data)
    return pca

def calc_mahalanobis_dist(feature_x1, feature_x2,var_a,var_b):
    feature_diff = feature_x1-feature_x2
    feature_diff = feature_diff * feature_diff
    if(var_a+var_b==0):
    	return 0;
    return feature_diff/(var_a+var_b)

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
	substracted = np.square(real_payload_map - payloads_map_mean)
	return np.divide(substracted, payloads_map_variances, out=np.zeros_like(substracted), where=payloads_map_variances!=0)

def predictAnomaly(weight, positive_threshold, negative_threshold):
	if weight < positive_threshold and weight > negative_threshold:
		return 0
	return 1

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
	# df = pd.read_csv(r'access_log_parsed.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
	# df.columns = ['index','destinationIP', 'timestamp', 'payload']

	df = pd.read_json('sqlinjections.json', lines=True)

	#Should have atleast 2 logs
	if len(df) >= 2:
		df = df[df['payload'].notnull()].copy()
		number_of_observations = len(df)

		# Vectorizing and calculating frequencies for payloads
		vectorized = np.vstack(df['payload'].apply(lambda x: vectorize_payload(x)))
		features_frequency = np.vstack([calc_frequency(x) for x in vectorized])

		compressed_data = compressed_PCA.transform(features_frequency)
		# print("==========Compressed Data==========")
		# print("Compressed Data Shape: ",compressed_data.shape)
		# print("Compressed Data: ",compressed_data)
		# print("Compressed explained_variance_ratio: ",compressed_PCA.explained_variance_ratio_)
		# print("Compressed singular_values: ",compressed_PCA.singular_values_)
		# print("Compressed explained_variance_ratio: ",compressed_PCA.explained_variance_ratio_.sum())
		# print("Cumsum: ",np.cumsum(compressed_PCA.explained_variance_ratio_))

		#Calculating meanshift for reduced features
		features_compressed_mean = calc_mean(compressed_data)
		mean_shift_compressed = np.transpose(compressed_data - features_compressed_mean)
		# print(mean_shift_compressed.shape)

		#Covariance Matrix for mean_shift_reduced
		# cov_mat_reduced = np.matmul(mean_shift_compressed, mean_shift_compressed.T)/(number_of_observations-1)
		# print(cov_mat_reduced.shape)
		# print(cov_mat_reduced)

		# Calculating mean of compressed data for covariances
		# compressed_data_mean = calc_mean(compressed_data)
		# covariances = format_covs(compressed_data, compressed_data_mean)

		# Constructing payload map using covariance matrix for reduced data
		real_payload_maps = []
		for x in compressed_data:
			real_payload_maps.append(construct_mdm(x, features_compressed_mean))
		real_payload_maps = np.array(real_payload_maps)
		print(real_payload_maps.shape)

		# Calculating thresholds from normal map mean and sd
		payloads_sd_sum = np.sum(payloads_map_sd)
		positive_threshold = 3*payloads_sd_sum
		negative_threshold = -3*payloads_sd_sum

		mean_sum = np.sum(payloads_map_mean)
		positive_threshold += mean_sum
		negative_threshold += mean_sum
		print("positive_threshold: ",positive_threshold)
		print("negative_threhold: ",negative_threshold)

		#Predicting anomalies
		count_anomaly=0
		count_already=0
		anomalies = []
		for idx, x in enumerate(real_payload_maps):
			weight = cal_weight(x, payloads_map_mean, payloads_map_vars)
			weight = np.sum(weight)
			anomaly = predictAnomaly(weight, positive_threshold, negative_threshold)
			anomalies.append(anomaly)
			#print("===================DETECTION RESULT===================")
			#print("Payload Weight: ",weight)
			#print("Anomaly Result: ", anomaly)
			# print(df.iloc[idx])
			if anomaly == 1:
				count_anomaly += 1
				# print("New Web Payload Found!")
			else:
				count_already += 1
				# print("Payload Already Exists!")

			# randomSleep = random.randint(2, 6)
			# print("Fetching next payload for analysis...")
			# time.sleep(randomSleep)
		df['anomaly'] = anomalies
		print("Number of anomalies: ",count_anomaly)
		print("Normal: ",count_already)
		print(df.head())

		# attacker_payloads = {}
		# for idx, row in df.iterrows():
		# 	if row['destinationIP'] in attacker_payloads.keys():
		# 		attacker_payloads[row['destinationIP']].append(row['payload'])
		# 	else:
		# 		attacker_payloads[row['destinationIP']] = [row['payload']]
		# 	print(attacker_payloads[row['destinationIP']])
		resumeAPScheduler()
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
