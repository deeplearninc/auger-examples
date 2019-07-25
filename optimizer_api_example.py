import requests

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

augerAPI = AugerAPI("https://app-staging.auger.ai")
augerAPI.login(email="email", password="password")


search_space = {
    "pyspark.ml.classification.RandomForestClassifier": {
        "maxDepth": {"bounds": [5, 20], "type": "int"},
        "maxBins": {"bounds": [32, 128], "type": "int"},
        "impurity": {"values": ["gini", "entropy"], "type": "categorical"},
        "numTrees": {"bounds": [20, 200], "type": "int"}
    },
    "pyspark.ml.classification.GBTClassifier": {
        "maxIter": {"bounds": [20, 200], "type": "int"},
        "maxBins": {"bounds": [32, 128], "type": "int"},        
        "stepSize": {"bounds": [0.1, 1.0], "type": "float"},
        "featureSubsetStrategy": {"values": ['auto', 'all', 'sqrt', 'log2'], "type": "categorical"}        
    }
}

augerAPI.create_trial_search(trials_total_count=100, search_space=search_space)

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

next_trials = augerAPI.continue_trial_search(trials_limit=4, trials_history=trials_history)
print(len(next_trials))

#Execute trials to get score, bigger is better (0.0..1.0)
for item in next_trials:
	item['score'] = 0.9
	item['evaluation_time'] = 1.0

next_trials = augerAPI.continue_trial_search(trials_limit=6, trials_history=next_trials)
print(len(next_trials))

