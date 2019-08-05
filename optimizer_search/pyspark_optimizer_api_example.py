import time
import copy
from auger_api import AugerAPI 

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("MLAugerSample").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# augerAPI = AugerAPI("https://app-staging.auger.ai")
# augerAPI.login(email="evgeny@auger.ai", password="bookes")

augerAPI = AugerAPI("https://app.auger.ai")
augerAPI.login(email="email", password="password")

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

trials_total_count = 20
augerAPI.create_trial_search(trials_total_count=trials_total_count, search_space=search_space)
next_trials = augerAPI.continue_trial_search(trials_limit=4)

data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

(training_data, test_data) = data.randomSplit([0.8, 0.2])

print("Start execute trials: %s"%next_trials)
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