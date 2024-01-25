# Unknown data handling

- [Unknown data handling](#unknown-data-handling)
  - [DaisyRec](#daisyrec)
    - [DaisyRec Q1](#daisyrec-q1)
    - [DaisyRec Q2](#daisyrec-q2)
    - [DaisyRec Q3](#daisyrec-q3)
    - [DaisyRec Summary](#daisyrec-summary)
  - [Elliot](#elliot)
    - [Elliot Q1](#elliot-q1)
    - [Elliot Q2](#elliot-q2)
    - [Elliot Q3](#elliot-q3)
    - [Elliot Summary](#elliot-summary)
  - [RecPack](#recpack)
    - [RecPack Q1](#recpack-q1)
    - [RecPack Q2](#recpack-q2)
    - [RecPack Q3](#recpack-q3)
    - [RecPack Summary](#recpack-summary)

For systems which use setting 3, the following describes how the RS handles the unseen user/item in the test set.

The following scenarios are considered

1. After the split is done, is there any checks on existence of user in test from train set
2. During the evaluation, how is the unknown user handled
3. Another other data handling methods

## DaisyRec

### DaisyRec Q1

As a recap, the follow code is used the split the train and test set.

```python
elif test_method == 'tsbr':
    split_idx = int(np.ceil(len(df) * (1 - test_size)))
    train_ids, test_ids = np.arange(split_idx), np.arange(split_idx, len(df))
```

Following from the split, the following [page](https://github.com/recsys-benchmark/DaisyRec-v2.0/blob/88b0a32faa1144033e969882781a8b0354958cf6/test.py) shows how the RS handles the test set.

```python
 ''' Train Test split '''
splitter = TestSplitter(config)
train_index, test_index = splitter.split(df)
train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()

''' get ground truth '''
# note that ur stands for user-rating but the RS uses item in its place
test_ur = get_ur(test_set) # ground truths for user-item in test set
test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config) # takes in the test user to 

test_dataset = CandidatesDataset(test_ucands) # creates the set for the model to perform prediction on
test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
preds = model.rank(test_loader)

results = calc_ranking_results(test_ur, preds, test_u, config) # perform evaluation
```

Expanding on the function that creates the set for the model to perform prediction on. We can conclude that unknown user or items are not dropped.

```python
def build_candidates_set(test_ur, train_ur, config, drop_past_inter=True):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_num : No. of all items
    cand_num : int, the number of candidates
    drop_past_inter : drop items already appeared in train set

    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    item_num = config['item_num']
    candidates_num = config['cand_num']

    test_ucands, test_u = [], []
    for u, r in test_ur.items(): # for each user id, item id
        sample_num = candidates_num - len(r) if len(r) <= candidates_num else 0
        if sample_num == 0:
            samples = np.random.choice(list(r), candidates_num)
        else:
            # NOTE: the positive item can be define purely by items in test+train or just test
            pos_items = list(r) + list(train_ur[u]) if drop_past_inter else list(r)
            # NOTE: negative item defined by all items that exist and not in positive sample set
            neg_items = np.setdiff1d(np.arange(item_num), pos_items)
            samples = np.random.choice(neg_items, size=sample_num) # sample num is just a hyper-param
            # NOTE: the test set contains negative samples and all positive items in the original test set
            samples = np.concatenate((samples, list(r)), axis=None)

        test_ucands.append([u, samples])
        test_u.append(u)
    
    return test_u, test_ucands
```

### DaisyRec Q2

Taking the **Popularity** metric as an example, the intersection of the prediction and ground truth is done first before the metric is computed. Irregardless of the user or item being known, the metric is still computed.
$$\frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}$$

```python
def Popularity(test_ur, pred_ur, test_u, item_pop):
    '''
    Abdollahpouri, Himan, et al. "The unfairness of popularity bias in recommendation." arXiv preprint arXiv:1907.13286 (2019).

    \frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}

    Parameters
    ----------
    test_ur : defaultdict(set)
        ground truths for user in test set
    pred_ur : np.array
        rank list for user in test set
    test_u : list
        the user in order from test set
    '''
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        i = np.intersect1d(pred, list(gt))
        if len(i):
            avg_pop = np.sum(item_pop[i]) / len(gt)
            res.append(avg_pop)
        else:
            res.append(0)

    return np.mean(res)

def Precision(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)): # for each user in the test set
        u = test_u[idx] # get the id of target user
        gt = test_ur[u] # find out ground truth of the target user
        pred = pred_ur[idx] # get the prediction of target user
        pre = np.in1d(pred, list(gt)).sum() / len(pred) # count of intersect div number of prediction 

        res.append(pre)

    return np.mean(res)
```

### DaisyRec Q3

Another observation on how the positive samples are defined.

```python
class Preprocessor(object):
        """
        Method of loading certain raw data
        Parameters
        ----------
        src : str, the name of dataset
        prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
        binary : boolean, whether to transform rating to binary label as CTR or not as Regression
        pos_threshold : float, if not None, treat rating larger than this threshold as positive sample
        level : str, which level to do with f'{N}core' or f'{N}filter' operation (it only works when prepro contains 'core' or 'filter')

        Returns
        -------
        df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
        """
def __reserve_pos(self, df):
    # set rating >= threshold as positive samples
    if self.pos_threshold is not None:
        df = df.query(f'rating >= {self.pos_threshold}').reset_index(drop=True)
    return df
```

Removal of duplicates. Repeated user,item interaction is dropped.

```python
def __remove_duplication(self, df):
    return df.drop_duplicates([self.uid_name, self.iid_name], keep='last', ignore_index=True)
```

Pre filtering of the dataset is also done before hand using either N-filter or Core filter

```python
elif self.prepro.endswith('filter'):
    pattern = re.compile(r'\d+')
    filter_num = int(pattern.findall(self.prepro)[0])

    tmp1 = df.groupby(['user'], as_index=False)['item'].count()
    tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
    tmp2 = df.groupby(['item'], as_index=False)['user'].count()
    tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
    df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
    if self.level == 'ui':    
        df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
    elif self.level == 'u':
        df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
    elif self.level == 'i':
        df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()        
    else:
        raise ValueError(f'Invalid level value: {self.level}')

    df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
    del tmp1, tmp2
    gc.collect()
elif self.prepro.endswith('core'):
    pattern = re.compile(r'\d+')
    core_num = int(pattern.findall(self.prepro)[0])

    if self.level == 'ui':
        user_inter_num = Counter(df[self.uid_name].values)
        item_inter_num = Counter(df[self.iid_name].values)
        while True:
            ban_users = self.__get_illegal_ids_by_inter_num(df, 'user', user_inter_num, core_num)
            ban_items = self.__get_illegal_ids_by_inter_num(df, 'item', item_inter_num, core_num)

            if len(ban_users) == 0 and len(ban_items) == 0:
                break

            dropped_inter = pd.Series(False, index=df.index)
            user_inter = df[self.uid_name]
            item_inter = df[self.iid_name]
            dropped_inter |= user_inter.isin(ban_users)
            dropped_inter |= item_inter.isin(ban_items)
            
            user_inter_num -= Counter(user_inter[dropped_inter].values)
            item_inter_num -= Counter(item_inter[dropped_inter].values)

            dropped_index = df.index[dropped_inter]
            df.drop(dropped_index, inplace=True)

    elif self.level == 'u':
        tmp = df.groupby(['user'], as_index=False)['item'].count()
        tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
        df = df.merge(tmp, on=['user'])
        df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
        df.drop(['cnt_item'], axis=1, inplace=True)
    elif self.level == 'i':
        tmp = df.groupby(['item'], as_index=False)['user'].count()
        tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp, on=['item'])
        df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
        df.drop(['cnt_user'], axis=1, inplace=True)
    else:
        raise ValueError(f'Invalid level value: {self.level}')

    gc.collect()
```

### DaisyRec Summary

1. During pre-processing all duplicated user item interaction is dropped
   1. Implication: a user who interacted multiple times with the item is ignored
2. During pre-processing, filtering of data can be done such that only users with sufficient item interactions are considered
3. During pre-processing all user and items are known where every unique user/item is obtained and stored through *__get_stats*
   1. This impacts the negative sampling later on since a 'future' item could be used as a negative sample. Since the item is not known at train time, the item will never be predicted. This will affect the prediction list used for evaluation.
4. Positive items can contain test+train items when creating set in *build_candidates_set*
   1. Positive items to be predicted are not purely from test set. We assume that past positive items are still positive.
5. Negative items are all items in universe that are not in positive set
6. The final test set will contain both negative and positive samples defined above
   1. The case of unknown user and item will still be used in prediction
7. Unknown users/item are still computed and depending on the rec-list provided, the score will vary

## Elliot

### Elliot Q1

The code below is a refresher on how the setting 3 split is done. A flag is created to filter test and train, where the flag acts as the global timeline. No other filtering is done to ensure that both user/item exist in train/test set.

```python
def splitting_passed_timestamp(self, d: pd.DataFrame, timestamp=1):
    tuple_list = []
    data = d.copy()
    data["test_flag"] = data.apply(lambda x: x["timestamp"] >= timestamp, axis=1)
    test = data[data["test_flag"] == True].drop(columns=["test_flag"]).reset_index(drop=True)
    train = data[data["test_flag"] == False].drop(columns=["test_flag"]).reset_index(drop=True)
    tuple_list.append((train, test))
    return tuple_list
```

### Elliot Q2

The cutoff is defined by the top-K items used to evaluate if there exists a cutoff. The relevance is defined by the test data being filtered by the relevance threshold.

```python
class Precision(BaseMetric):
    def __init__(self, recommendations, config, params, eval_objects):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects)
        self._cutoff = self._evaluation_objects.cutoff
        self._relevance = self._evaluation_objects.relevance.binary_relevance

    def __user_precision(self, user_recommendations, user, cutoff):
        """
        Per User Precision
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        # For each item predicted for the target user, check if it is in the test set
        # If in test set add 1 to the count. Find the overall number of relevant items and divide by cutoff
        return sum([self._relevance.get_rel(user, i) for i, _ in user_recommendations[:cutoff]]) / cutoff

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Precision
        """
        return {u: self.__user_precision(u_r, u, self._cutoff)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}
```

### Elliot Q3

Code definition on how a negative sample is derived and computed against a positive sample. This is different from [above](#daisyrec)

```python
class AUC(BaseMetric):
    def __init__(self, recommendations, config, params, eval_objects):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects)
        self._cutoff = self._evaluation_objects.cutoff
        self._relevance = self._evaluation_objects.relevance.binary_relevance
        self._num_items = self._evaluation_objects.num_items

    @staticmethod
    def __user_auc(user_recommendations, user_relevant_items, num_items, train_size):
        """
        Per User Computation of AUC values
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :param num_items: overall number of items considered in the training set
        :param train_size: length of the user profile
        :return: the list of the AUC values per each test item
        """
        neg_num = num_items - train_size - len(user_relevant_items) + 1
        pos_ranks = [r for r, (i, _) in enumerate(user_recommendations) if i in user_relevant_items]
        return [(neg_num - r_r + p_r) / (neg_num) for p_r, r_r in enumerate(pos_ranks)]
```

### Elliot Summary

1. Pre filtering by rating relevance
   1. Drops an interaction if the rating is below a constant defined, global average rating, user average rating
2. Pre filtering by k core
3. Pre filtering and selecting of only cold users
4. The negative items in Elliot is computed based on hyper-param definition and is just a numeric figure unlike DaisyRec which tries to sample from possible items
5. Unknown users/item are still computed and depending on the rec-list provided, the score will vary

## RecPack

### RecPack Q1

The function below splits the train and test set when called by [scenario.Timed](https://github.com/LienM/recpack/blob/d42997736d0d5dfda5127a7690c732465aa03df3/recpack/scenarios/timed.py#L17).

```python
class TimestampSplitter(Splitter):
    """Split data so that the first return value contains interactions in ``[t-delta_in, t[``,
    and the second those in ``[t, t+delta_out[``.

    If ``delta_in`` or ``delta_out`` are omitted, they are assumed to have a value of infinity.
    A user can occur in both return values.

    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    :param delta_out: Seconds past t. Upper bound on the timestamp
        of interactions in the second return value. Defaults to None (infinity).
    :type delta_out: int, optional
    :param delta_in: Seconds before t. Lower bound on the timestamp
        of interactions in the first return value. Defaults to None (infinity).
    :type delta_in: int, optional
    """

    def __init__(self, t, delta_out=None, delta_in=None):
        super().__init__()
        self.t = t
        self.delta_out = delta_out
        self.delta_in = delta_in

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data so that ``data_in`` contains interactions in ``[t-delta_in, t[``,
        and ``data_out`` those in ``[t, t+delta_out[``.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the ``data_in`` and ``data_out`` matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        assert data.has_timestamps

        if self.delta_in is None:
            # timestamp < t
            data_in = data.timestamps_lt(self.t)
        else:
            # t-delta_in =< timestamp < t
            data_in = data.timestamps_lt(self.t).timestamps_gte(self.t - self.delta_in)

        if self.delta_out is None:
            # timestamp >= t
            data_out = data.timestamps_gte(self.t)
        else:
            # timestamp >= t and timestamp < t + delta_out
            data_out = data.timestamps_gte(self.t).timestamps_lt(self.t + self.delta_out)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out
```

### RecPack Q2

The demo usecase of the algorithm can be found [here](https://github.com/LienM/recpack/blob/d42997736d0d5dfda5127a7690c732465aa03df3/examples/Demo.ipynb)

Uses a pipeline to train, test and validate. The main method called is **pipeline.run**

```python
def run(self):
    """Runs the pipeline."""
    for algorithm_entry in tqdm(self.algorithm_entries):
        # Check whether we need to optimize hyperparameters
        if algorithm_entry.optimise:
            params = self._optimise_hyperparameters(algorithm_entry)
        else:
            params = algorithm_entry.params

        algorithm = ALGORITHM_REGISTRY.get(algorithm_entry.name)(**params)
        if isinstance(algorithm, TorchMLAlgorithm):
            self._train(algorithm, self.validation_training_data) # training with validation
        else:
            self._train(algorithm, self.full_training_data) # training without validation
        # Make predictions
        X_pred = self._predict_and_postprocess(algorithm, self.test_data_in)

        for metric_entry in self.metric_entries:
            metric_cls = METRIC_REGISTRY.get(metric_entry.name)
            if metric_entry.K is not None:
                metric = metric_cls(K=metric_entry.K)
            else:
                metric = metric_cls()
            metric.calculate(self.test_data_out.binary_values, X_pred)
            self._metric_acc.add(metric, algorithm.identifier, metric.name)
```

How each metric is computed would are as follow

```python
class PrecisionK(ListwiseMetricK):
    """Computes the fraction of top-K recommendations that correspond
    to true interactions.

    Different from the definition for information retrieval
    a recommendation algorithm is expected to always return K items
    when the Top-K recommendations are requested.
    When fewer than K items received scores, these are considered a miss.
    As such recommending fewer items is not beneficial for a
    recommendation algorithm.

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1

        scores = scores.tocsr()

        self.scores_ = csr_matrix(scores.sum(axis=1)) / self.K

        return
```

### RecPack Q3

With regards to positive and negative sampling

```python
class PositiveNegativeSampler(Sampler):
    """Samples linked positive and negative interactions for users.

    Provides a :meth:`sample` method that samples positives and negatives.
    Positives are sampled uniformly from all positive interactions.
    Negative samples are sampled either based on a uniform distribution
    or a unigram distribution.

    The uniform distrbution makes it so each item has the same probability to
    be selected as negative.
    With the unigram distribution, items are sampled according to their weighted
    popularity.
    """
```

### RecPack Summary

1. Variety of pre-processing filters that are similar in nature to the other 2 RS
   1. MinUsersPerItem - Require that a minimum number of users has interacted with an item
   2. NMostPopular - Retain only the N most popular items.
   3. MinItemsPerUser - Require that a user has interacted with a minimum number of items.
   4. Deduplicate - Deduplicate entries with the same user and item. Removes all duplicated user-item pair
2. As for positive negative sampling, for each positive sample a negative sample is generated
3. Unknown user/item are still computed and taken as a miss
