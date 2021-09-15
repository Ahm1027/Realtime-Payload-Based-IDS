import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


# model



def load_training_data():
    df= pd.read_csv('/content/drive/MyDrive/http_requests.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    df = shuffle(df)
    df['payload'] = df['payload'].fillna(0)
    df['payload'] = df['payload'].apply(lambda x: str(x))
    df.drop(columns=['index', 'method', 'url', 'protocol', 'userAgent', 'pragma', 'cacheControl', 'accept', 'acceptEncoding', 'acceptCharset', 'acceptLanguage', 'host', 'connection', 'contentLength', 'contentType', 'cookie'], inplace=True)
    return df

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

def calc_cov_mat(features_frequency, features_mean):
    mean_shift = np.transpose(np.subtract(features_frequency, features_mean))    
    return np.cov(mean_shift), mean_shift

def calc_eig_pairs(cov_mat):
    eig_val, eig_vec = np.linalg.eigh(cov_mat)
    eig_pairs = [(np.abs(eig_val[x]), eig_vec[:,x], x) for x in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    for i in eig_pairs:
        print(i[0])

def sum_cumulative_energy(pairs):
    THRESHOLD = 100e-6
    sum = 0
    for x in pairs:
        sum += x[0] if x[0] > THRESHOLD else 0
    return sum

def cumulative_constant(cumulative_sum, pairs, cumulative_threshold=99.5):
    current_sum, curr_cumsum = 0, 0
    i = 0
    pc = []
    while i != len(pairs)-1 and curr_cumsum < cumulative_threshold:
        # print(pairs[i][2])
        current_sum += pairs[i][0]
        curr_cumsum = (current_sum/cumulative_sum)*100
        print("Current Sum: %f, Curr cumsum: %f" % (current_sum, curr_cumsum))
        pc.append(pairs[i][2])
        i += 1
    return i, pc

def cal_pca(features_data):
    pca = PCA(30)
    pca.fit_transform(features_data)
    return pca


def cal_covariance(feature):
    feature_vector = np.array([feature])
    result = np.matmul(feature_vector, np.transpose(feature_vector))
    return result[0][0]

def format_freq(compressed_data):
    features = np.transpose(compressed_data)
    covs = np.array([], dtype=np.float16)
    return np.append(covs, [cal_covariance(x) for x in features])

def cal_covariance(feature):
    feature_vector = np.array([feature, feature])
    result = np.matmul(feature_vector, np.transpose(feature_vector))
    return result[0][0]

def calc_mahalanobis_dist(feature_x1, feature_x2, cov_x1, cov_x2):
    feature_diff = feature_x1-feature_x2
    feature_diff = feature_diff*np.transpose(feature_diff)
    if cov_x1+cov_x2 == 0:
        return 0
    return feature_diff/(cov_x1+cov_x2)
  
def construct_mdm(features,covariances):
    mdm = []
    i=0
    for index, feature1 in enumerate(features):
        feature_map = []
        for index2, feature2 in enumerate(features):
            feature_map.append(calc_mahalanobis_dist(feature1, feature2, covariances[index], covariances[index2]))
        mdm.append(feature_map)
    return np.array(mdm)


def training():
    df=load_training_data()
    vectors = np.array(df['payload'].apply(lambda x: vectorize_payload(x)))
    results = [calc_frequency(x) for x in vectors]
    features_frequency = np.vstack(results)
    del results
    features_mean = calc_mean(features_frequency)
    cov_mat, mean_shift = calc_cov_mat(features_frequency, features_mean)

    eig_pairs=calc_eig_pairs(cov_mat)
    cumulative_sum = sum_cumulative_energy(eig_pairs)
    num_of_PC, cumsum = cumulative_constant(cumulative_sum, eig_pairs)

    pca = cal_pca(features_frequency)
    compressed_SVD = TruncatedSVD(30)
    compressed_data = compressed_SVD.fit_transform(features_frequency)

    features_reduced_mean = calc_mean(compressed_data)
    features_reduced_sd = calc_sd(compressed_data)
    covariances = format_freq(compressed_data)

    payload_map = np.array([construct_mdm(x,covariances) for x in compressed_data[:10000]])
    payloads_map_mean = calc_mean(payload_map)
    payloads_map_sd= calc_sd(payload_map)

    payloads_map_variances = np.square(payloads_map_sd)

    return payloads_map_mean,payloads_map_variances,compressed_data


def construct_payload_map(payload):
  vectorized = vectorize_payload(payload)
  payload_frequency = calc_frequency(vectorized)
  payload_mean = calc_mean(payload_frequency)
  payload_sd = np.std(payload_frequency)
  mean_shift = np.transpose(np.subtract(payload_frequency, payload_mean))    
  payload_cov_mat = np.cov(mean_shift)
  print(payload_cov_mat)




payloads_map_mean,payloads_map_variances,compressed_data=training()
real_payload_map =compressed_data[130000]

def cal_weight(real_payload_map):
  return (np.square(real_payload_map - payloads_map_mean)/payloads_map_variances)

payload_weight = cal_weight(real_payload_map)

weight_sum=np.sum(payload_weight)
