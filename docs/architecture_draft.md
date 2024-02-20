# FYP Draft 1

- [FYP Draft 1](#fyp-draft-1)
- [Architecture](#architecture)
- [Assumptions](#assumptions)
- [Procedures](#procedures)
  - [Handling of unknown user](#handling-of-unknown-user)
- [Evaluation flow](#evaluation-flow)
- [Loading dataset](#loading-dataset)
  - [Loading without error](#loading-without-error)
  - [Handling of dataset without timeline](#handling-of-dataset-without-timeline)
- [Evaluation mechanism](#evaluation-mechanism)
- [Splitting Dataset](#splitting-dataset)
  - [Class diagram](#class-diagram)

# Architecture

Proposed name: StreamSight

# Assumptions

- Dataset input must come with timestamp for partitioning
- Models used must support provided API by SteamSight

# Procedures

## Handling of unknown user

Definition of Unknown: User appears in the test set but is not seen before in the train set.

Since these unknown users cannot be evaluated, they can first be

1. Ignored
2. Be always assigned a default value such as the most popular item

# Evaluation flow

The configuration file should be stored as a yaml file and should contain the following details

- recommender_type: [KNN]
- split_type: [global_timeline|single_timeline|leave_one_out|random]
- train_data_sampled: bool
- test_data_sampled: bool
- dataset_url: str
- window_size: int
- num_window_per_set: int


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

## Loading without error

```plantuml
@startuml Loading without error

[-> Evaluator : Provide url
Evaluator -> Loader : load(url)
activate Loader
Loader -> Dataset

activate Dataset
Dataset -> Dataset : Download from url
Dataset -> Dataset : Validate timestamp
Dataset -> Dataset : Format into Dataframe
Dataset --> Loader
deactivate Dataset

Loader -> Loader : Sort interaction
Loader -> Loader : Filter

Loader --> Evaluator
deactivate Loader
@enduml
```

## Handling of dataset without timeline
```plantuml
@startuml Loading without timeline
title Handling dataset without timeline
Evaluator -> Loader : load(url)
activate Loader
Loader -> Dataset

activate Dataset
Dataset -> Dataset : Download from url
Dataset -> Dataset : Validate timestamp
alt valid timestamp
    Dataset -> Dataset : Format into Dataframe
    Dataset --> Loader --

    Loader -> Loader : Filtering

    Loader --> Evaluator --
else invalid timestamp
    Dataset -[hidden]> Loader ++
    activate Dataset

    Dataset -> Loader --: Raise error
    Loader -> Evaluator --: Raise InvalidDataset
end

@enduml
```

---

# Evaluation mechanism

Following from the paper. If the evaluation metric can happen at each window and an aggregated level,
then for 
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

# Splitting Dataset

```plantuml
@startuml Splitting data
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

