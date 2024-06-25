# FYP Draft 1

- [FYP Draft 1](#fyp-draft-1)
- [Architecture](#architecture)
- [Assumptions](#assumptions)
- [Questions/ Trailing solutions](#questions-trailing-solutions)
- [Documentation](#documentation)
- [Procedures](#procedures)
- [Evaluation flow](#evaluation-flow)
- [Loading dataset](#loading-dataset)
  - [Pipeline overview of loading dataset](#pipeline-overview-of-loading-dataset)
- [Splitting Dataset](#splitting-dataset)
- [Evaluation mechanism](#evaluation-mechanism)
  - [Class diagram](#class-diagram)

# Architecture

Proposed name: StreamSight

# Assumptions

- Dataset input must come with timestamp for partitioning
- Models used must support provided API by SteamSight

# Questions/ Trailing solutions

What about unknown users that appear in the test set? Meaning they are not in
training set, but in test set
- We have 2 settings, to ignore these users during evaluation
- or to set a default value for these users such as having the most popular item


# Documentation

The docstring used in the code base follows reST formatting. This is because
it is compatible with Sphinx for documentation for easy documentation generation
along the way.

# Procedures

# Evaluation flow

The flow of the 

```plantuml
@startuml Overall flow

actor Actor as user

user -> Evaluator : start(config_file_path)

Evaluator -> Loader : Calls for dataset
Loader --> Evaluator : Return dataset

Evaluator -> Splitter : split(window_size,num_window)
Splitter -->  Evaluator : Return train_set_list,test_set_list

loop each window set
Evaluator -> Recommender : Calls fit(train_set) and pred(test_set)
Recommender --> Evaluator : Return predict(test_set) output
end

Evaluator --> user

@enduml
```

---

# Loading dataset

## Pipeline overview of loading dataset

Pipeline builder provides the API to add the dataset of interest to the pipeline this. By calling `pipeline_builder.set_dataset(dataset)` we allow the user to specify the dataset to load. The dataset specified can be the class or a string argument.

Once the pipeline builder has the specified arguments, calling `run()` on the pipeline will cause the dataset class to be instantiated and subsequently loaded.

```plantuml
@startuml
title Pipeline interaction with loader
activate Pipeline
Pipeline -> Pipeline ++: run()
Pipeline -> Dataset ++
Dataset -> Dataset ++ : load()
Dataset -> Dataset: fetch_dataset()
note right: called to download \n or load from storage
Dataset -> Dataset -- : _dataframe_to_matrix()
Dataset --> Pipeline -- : dataset
Pipeline -[hidden]-> Pipeline --
@enduml
```

---

# Splitting Dataset

We split the dataset into 2 broad category. (1) Full training dataset and (2) test data. For (1) we further split it into validation train and test data if the user specifies for validation set to be created. The validation train and test is a subset of (1) and for the final model, the full training dataset will be used.

We assume that all datasets used will contain a timestamp. Instead of sorting the entire dataset then splitting by some index, we will simply use pandas build-in tools to aid us.

We take an example of splitting based on a single global timeline. Given a timestamp `t` we will filter for all rows that have the timestamp less than `t` to be part of the in-set, or (1). And for all items greater equal to `t`, it will be contained in (2).

We provide the capability to indicate a `delta_in` and `delta_out` such that the user can indicate the extent of time range for the dataset to be used. If not specified, it will be simply as explained above.


Now we take the example of a sliding global timeline. The i

Note that we can have 2 types of spliitng for the global timeline

1. To restrict the slide the entire training set window forward such that\
    the number of windows per set is preserved
2. To provide the test set as the training set for the next iteration

```plantuml
@startuml Splitting data
title Splitting data
activate Evaluator
Evaluator -> Splitter ++ : split(split_type,window_size,num_window)
Splitter -> Splitter : Determine split type
Splitter -> Splitter : Check valid window size
Splitter -> Splitter : Check valid window number per set

alt No error raised
    loop each window set
        Splitter -> Dataset : Split dataset
        Dataset --> Splitter : train_set,test_set
    end

    Splitter -->Evaluator --: list of train and test set
    
else Error raised
    Splitter -[hidden]> Dataset
    activate Splitter
    Splitter -> Evaluator --: Raise Error 
end

@enduml
```

---

# Evaluation mechanism

Following from the paper. If the evaluation metric can happen at each window
and an aggregated level, then for

1. User level
    - Each window, and each user, we evaluate the result
2. Aggregated level
    - Each window for all users aggregated

```plantuml
@startuml Evaluation
activate Evaluator

loop train_set,test_set   

    Evaluator  -> Recommender ++: train_set
    Recommender -> Recommender : train(train_set)
    Recommender --> Evaluator

    Evaluator -> Recommender : test_set
    Recommender -> Recommender : pred(test_set)
    Recommender --> Evaluator --: prediction

    loop each metric
    
        loop each user
            Evaluator -> Metric ++: prediction,ground_truth
            Metric --> Evaluator --: Results
        end
        
        Evaluator -> Metric ++: prediction,ground_truth
        Metric --> Evaluator --: Results
    end 

end
@enduml
```

---

## Class diagram

```plantuml
@startuml Class diagram

class Evaluator {
    + load_dataset()
    + split_dataset()
    + Evaluate_model(model,train_set_list,test_set_list)
    - train_model(model,train_set_list)
    - test_model(model,test_set_list)
}
class Loader {
    + url : str
    + dataset : Dataset
    + load(url) : Dataset
}
class Splitter {
    + train_set_list : List
    + test_set_list : List
    + window_size : int
    + num_window_per_set : int
    + split : SplitType 
    + split(dataset : Dataset) : List, List
}
enum SplitType <<Enumeration>> {
    RANDOM
    GLOBAL_TIMELINE
    LEAVE_ONE_OUT
}
abstract Recommender <<Abstract>> {
    + fit(train_set)
    + predict(test_set)
}
class UserKNN {
    + fit(train_set)
    + predict(user : int)
}
class Dataset {
    + num_user : int
    + num_item : int
    + dataset : Dataframe
    + get_users() : List
    + get_items() : List
}

hide <<Enumeration>> circle

Recommender <|-- UserKNN
Loader --o Dataset
Splitter --o Dataset
Splitter --> SplitType
Evaluator -- Loader
Evaluator -- Splitter
Evaluator "1" -- "*" Recommender
@enduml
```

