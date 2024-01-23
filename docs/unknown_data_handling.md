# Unknown data handling

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

Taking the **Popularity** metric as an example, the intersection of the prediction and ground truth is done first before the metric is computed.

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

## Elliot

## Rec
