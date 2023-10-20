import xgboost as xgb
from xgboost import plot_importance, plot_tree


##### Step 3.a
'''
Implement an XGBoost training function.  Called from utilities/xgb_utils.py.

This should be verey similar to the how training is done in the LTR toy program.

:param str xgb_train_data: The file path to the training data you should train with
:param int num_rounds: The number of rounds the training process should undertake before terminating.
:param dictionary xgb_params The XGBoost configuration parameters, such as the objective function, e.g. {'objective': 'reg:logistic'} 
'''
def train(xgb_train_data, num_rounds=5, xgb_params=None ):
    print("Training XG Boost")
    # Do the training.  NOTE: in this toy example we did not use any hold out data
    # model = bst.get_dump(fmap=feat_map_file.name, dump_format='json')
    # dtrain = xgb.DMatrix(xgb_train_data)
    dtrain = xgb.DMatrix(f'{xgb_train_data}?format=libsvm')
    bst = xgb.train(xgb_params, dtrain, num_rounds)
    return bst

##### Step 3.b:
'''
We need to log/extract features for our training data set.  This is a critical step in the 
LTR process where we request features (from the featureset we definied) of all of the documents that match a given query from OpenSearch.
For instance, we might get back the TF-IDF score for a document given our query.

Called from ltr_utils.py

Unlike in the toy LTR example where we issued a single query to OpenSearch per query-document pair, you should be able to retrieve 
all features for all documents in a single query.  See the course content for more details. 

:param str query: The user query to run
:param array doc_ids: An array/list of doc ids that users have clicked on for this query. 
:param str click_prior_query: A query string representing what documents were clicked on in the past.  Not used until the 'Using Prior Query History' section of the project.
:param str featureset_name: The name of the featureset we should use to extract features from OpenSearch with
:param str ltr_store_name: The name of the LTR store we are using to extract features from
:param int size: The number of results to return in the search result set
:param string terms_field: The name of the field to filter our doc_ids on 
'''
def create_feature_log_query(query, doc_ids, click_prior_query, featureset_name, ltr_store_name, size=200, terms_field="_id"):
    # print("IMPLEMENT ME: create_feature_log_query with proper LTR syntax")
    # return {
    #     'size': size,
    #     'query': {
    #         'bool': {
    #             "filter": [  # use a filter so that we don't actually score anything
    #                 {
    #                     "terms": {
    #                         terms_field: doc_ids
    #                     }
    #                 }
    #             ]
    #         }
    #     }
    # }
    return {
        'size': size,
        'query': {
            'bool': {
                "filter": [
                    {
                        "terms": {
                            terms_field: doc_ids
                        }
                    },
                    {
                        "sltr": {
                            "_name": "logged_featureset",
                            "featureset": featureset_name,
                            "store": ltr_store_name,
                            "params": {
                                "keywords": query
                            }
                        }
                    }
                ]
            }
        },
        "ext": {
            "ltr_log": {
                "log_specs": {
                    "name": "log_entry",
                    "named_query": "logged_featureset"
                }
            }
        }
    }


##### Step 4.e:
'''
Modify the query_obj to add a `rescore` entry that uses the baseline query, the LTR information and the rescore window to 
create a new query that actually does the rescoring using our LTR model.
Called from ltr_utils.py

:param str user_query: The user query to run
:param dictionary query_obj: The query object to be submitted to OpenSearch to execute LTR rescoring. Modify this object to add the `rescore` entry with your rescoring query.
:param str click_prior_query: A query string representing what documents were clicked on in the past.  Not used until the 'Using Prior Query History' section of the project.
:param str featureset_name: The name of the featureset we should use to extract features from OpenSearch with
:param str ltr_store_name: The name of the LTR store we are using to extract features from
:param int rescore_size: The number of results to rescore
:param float main_query_weight: A float indicating how much weight to give results that match in the original query
:param float rewcore_query_weight: A float indicating how much weight to give results that match in the rescored query
'''
def create_rescore_ltr_query(user_query: str, query_obj, click_prior_query: str, ltr_model_name: str,
                             ltr_store_name: str,
                             rescore_size=500, main_query_weight=1, rescore_query_weight=2):
    # print("IMPLEMENT ME: create_rescore_ltr_query")
    # query_obj["rescore"] = { ... }
    query_obj["rescore"] = {
        "window_size": rescore_size,
        "query": {
            "rescore_query": {
                "sltr": {
                    "params": {
                        "keywords": user_query,
                        "skus": user_query.split(),
                        "click_prior_query": click_prior_query
                    },
                    "model": ltr_model_name,
                    "store": ltr_store_name
                }
            },
            "score_mode": "total",
            "query_weight": main_query_weight,
            "rescore_query_weight": rescore_query_weight
        }
    }


##### Step Extract LTR Logged Features:
'''
Using the hits object (e.g. response['hits']['hits']) returned by OpenSearch, iterate through the results
and extract the features into a data frame.

:param array hits: The array of hits returned by executing the feature_log_query object against an OpenSearch instance
:param int query_id: The id of the current query we are processing.
'''
def extract_logged_features(hits, query_id):
    import numpy as np
    import pandas as pd
    # print("IMPLEMENT ME: __log_ltr_query_features: Extract log features out of the LTR:EXT response and place in a data frame")
    feature_results = {}
    feature_results["doc_id"] = []  # capture the doc id so we can join later
    feature_results["query_id"] = []  # ^^^
    feature_results["sku"] = []
    feature_results["name_match"] = []
    rng = np.random.default_rng(12345)

    for (idx,hit) in enumerate(hits):
        features = hit['fields']['_ltrlog'][0]['log_entry']
        feature_results["doc_id"].append(hit['_id'])
        feature_results["sku"].append(int(hit['_source']['sku'][0]))
        feature_results["query_id"].append(int(query_id))
        for feat_idx, feature in enumerate(features):
            feat_name = feature.get('name')
            feat_val = feature.get('value', 0)
            feat_vals = feature_results.get(feat_name)
            if feat_vals is None:
                feat_vals = []
                feature_results[feat_name] = feat_vals
            feat_vals.append(feat_val)

    frame = pd.DataFrame(feature_results)
    return frame.astype({'doc_id': 'int64', 'query_id': 'int64', 'sku': 'int64'})