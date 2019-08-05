import requests

#https://app-staging.auger.ai/api/v1/docs
class AugerAPI(object):
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
		#print(self.token)

	def create_trial_search(self, trials_total_count, search_space):
		response = self.call_api(requests.post, "trial_searches", 
			{'trials_total_count': trials_total_count, 'search_space': search_space,
					'dataset_metafeatures':{'test': 123}})
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

	def continue_trial_search(self, trials_limit, trials_history=[]):

		if not trials_history:
			trials_history = [{
		      "uid": "83D2FE5F2B0A42F",
		      "error": None,
		      "score": 0.43478260869565216,
		      "budget": None,
		      "ratio": 1.0,
		      "evaluation_time": 0.40130186080932617,
		      "algorithm_name": "sklearn.ensemble.RandomForestClassifier",
		      "algorithm_params": {
		        "bootstrap": True,
		        "max_features": 0.7951475142804721,
		        "min_samples_leaf": 13,
		        "min_samples_split": 18,
		        "n_estimators": 219,
		        "n_jobs": 1
		      }
		    }]

		return self.call_api(requests.patch, "trial_searches/%s"%self.search_id, 
			{'trials_limit': trials_limit, 'trials_history': trials_history}).get('next_trials', [])

	@staticmethod		 
	def create_object_by_class(full_name, args):
	    import importlib

	    module_name, class_name = full_name.rsplit('.', 1)
	    cls = getattr(importlib.import_module(module_name), class_name)
	    return cls(**args)
