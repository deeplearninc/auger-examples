import requests
import time
import copy

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("MLAugerSample").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

#https://app-staging.auger.ai/api/v1/docs
class AugerAPI:
	def __init__(self, url):
		self.url = url
		self.token = None

	def call_api(self, rest_method, api_command, data):
		if self.token:
			data['token'] = self.token

		response = rest_method(self.url + "/api/v1/" + api_command, json=data).json()
		return response.get('data', {})

	def login(self, email, password):
		response = self.call_api(requests.post, "tokens", {'email': email, 'password': password})
		self.token = response.get('token')
		print(self.token)

	def create_trial_search(self, trials_total_count, search_space):
		response = self.call_api(requests.post, "trial_searches", 
			{'trials_total_count': trials_total_count, 'search_space': search_space,
					'dataset_metafeatures':{'test': 1}})
		self.search_id = response.get('id')

		status = response.get('status')
		print("Search ID: %s, status: %s"%(self.search_id, status))

		while status != 'done' and status != 'error':
			response = self.call_api(requests.get, "trial_searches/%s"%self.search_id, {})

			status = response.get('status')
			print("Search ID: %s, status: %s"%(self.search_id, status))

		if status == 'error':
			print("Error when create auger object:%s"%response)
			exit(1)

	def continue_trial_search(self, trials_limit, trials_history):
		 return self.call_api(requests.patch, "trial_searches/%s"%self.search_id, 
			{'trials_limit': trials_limit, 'trials_history': trials_history}).get('next_trials', [])

	@staticmethod		 
	def create_object_by_class(full_name, args):
	    import importlib

	    module_name, class_name = full_name.rsplit('.', 1)
	    cls = getattr(importlib.import_module(module_name), class_name)
	    return cls(**args)

augerAPI = AugerAPI("https://app-staging.auger.ai")
augerAPI.login(email="evgeny@auger.ai", password="bookes")

search_space = {
    "pyspark.ml.classification.RandomForestClassifier": {
        "maxDepth": {"bounds": [5, 20], "type": "int"},
        "maxBins": {"bounds": [4, 16], "type": "int"},
        "impurity": {"values": ["gini", "entropy"], "type": "categorical"},
        "numTrees": {"bounds": [4, 16], "type": "int"}
    },
    "pyspark.ml.classification.GBTClassifier": {
        "maxIter": {"bounds": [4, 16], "type": "int"},
        "maxBins": {"bounds": [4, 16], "type": "int"},        
        "stepSize": {"bounds": [0.1, 1.0], "type": "float"},
        "featureSubsetStrategy": {"values": ['auto', 'all', 'sqrt', 'log2'], "type": "categorical"}        
    }
}

trials_history = [
	{
	  "score":0.4347826086956522,
	  "evaluation_time":0.4013018608093262,
	  "algorithm_name":"pyspark.ml.classification.RandomForestClassifier",
	  "algorithm_params":{
	    "maxDepth":15,
	    "maxBins":40,
	    "impurity":"gini",
	    "numTrees":100
	  }
	}
]

trials_total_count = 20
augerAPI.create_trial_search(trials_total_count=trials_total_count, search_space=search_space)
next_trials = augerAPI.continue_trial_search(trials_limit=4, trials_history=trials_history)

data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

(training_data, test_data) = data.randomSplit([0.8, 0.2])

trials_history = []
while len(trials_history) < trials_total_count:
	#Execute trials to get score, bigger is better (0.0..1.0)
	#It may be run in parallel
	for trial in next_trials:
		algo_params = copy.deepcopy(trial.get('algorithm_params'))
		algo_params['labelCol'] = "indexedLabel"
		algo_params['featuresCol'] = "indexedFeatures"

		ml_algo = AugerAPI.create_object_by_class(trial.get('algorithm_name'), algo_params)
		pipeline = Pipeline(stages=[labelIndexer, featureIndexer, ml_algo])

		start_fit_time = time.time()
		ml_model = pipeline.fit(training_data)

		history_item = copy.deepcopy(trial)
		history_item['evaluation_time'] = time.time() - start_fit_time

		predictions = ml_model.transform(test_data)
		evaluator = MulticlassClassificationEvaluator(
		    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

		history_item['score'] = evaluator.evaluate(predictions)

		print("Executed trial: %s"%history_item)
		trials_history.append(history_item)

	next_trials = augerAPI.continue_trial_search(trials_limit=4, trials_history=trials_history)
	
trials_history.sort(key=lambda t: t['score'], reverse=True)

print("Best trial: %s"%trials_history[0])
spark.stop()