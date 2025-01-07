import logging
from warnings import warn

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

from streamsight.algorithms.itemknn import ItemKNN
from streamsight.matrix import InteractionMatrix
from streamsight.utils.util import add_rows_to_csr_matrix

logger = logging.getLogger(__name__)


class ItemKNNIncremental(ItemKNN):
    """Incremental version of ItemKNN algorithm.

    This class extends the ItemKNN algorithm to allow for incremental updates
    to the model. The incremental updates are done by updating the historical
    data with the new data by appending the new data to the historical data.
    """

    def __init__(self, K=10, pad_with_popularity=True):
        super().__init__(K=K)
        self.pad_with_popularity = pad_with_popularity
        self.training_data: csr_matrix = None

    def append_training_data(self, X: csr_matrix):
        """Append a new interaction matrix to the historical data.

        :param X: The new interaction matrix
        :type X: csr_matrix
        """
        if self.training_data is None:
            return
        X_prev: csr_matrix = self.training_data.copy()
        new_num_rows = max(X_prev.shape[0], X.shape[0])
        new_num_cols = max(X_prev.shape[1], X.shape[1])
        # Pad the previous matrix
        if X_prev.shape[0] < new_num_rows:  # Pad rows
            row_padding = csr_matrix((new_num_rows - X_prev.shape[0], X_prev.shape[1]))
            X_prev = vstack([X_prev, row_padding])
        if X_prev.shape[1] < new_num_cols:  # Pad columns
            col_padding = csr_matrix((X_prev.shape[0], new_num_cols - X_prev.shape[1]))
            X_prev = hstack([X_prev, col_padding])

        # Pad the current matrix
        if X.shape[0] < new_num_rows:  # Pad rows
            row_padding = csr_matrix((new_num_rows - X.shape[0], X.shape[1]))
            X = vstack([X, row_padding])
        if X.shape[1] < new_num_cols:  # Pad columns
            col_padding = csr_matrix((X.shape[0], new_num_cols - X.shape[1]))
            X = hstack([X, col_padding])

        # Merge data
        self.training_data = X_prev + X


    def _fit(self, X: csr_matrix) -> "ItemKNNIncremental":
        """Fit a cosine similarity matrix from item to item."""
        if self.training_data is None:
            self.training_data = X.copy()
        else:
            self.append_training_data(X)
        print("Training data: ", self.training_data.toarray())
        super()._fit(self.training_data)
        return self
    
    def _predict(self, X: csr_matrix, predict_im: InteractionMatrix) -> csr_matrix:
        """Predict the K most similar items for each item using the latest data."""
        X_pred = super()._predict(self.training_data)
        print("In ItemKNNIncremental _predict: ", X_pred.toarray())
        # ID indexing starts at 0, so max_id + 1 is the number of unique IDs
        max_user_id = predict_im.max_user_id + 1
        print("Max user ID: ", max_user_id)
        max_item_id = predict_im.max_item_id + 1
        print("Max item ID: ", max_item_id)
        intended_shape = (
            max(max_user_id, X.shape[0]),
            max(max_item_id, X.shape[1]),
        )
        print("X.shape: ", X.shape)
        print("Intended shape: ", intended_shape)

        predict_frame = predict_im._df
        print("Predict frame: ", predict_frame)

        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = X_pred.shape
        print("Known user ID: ", known_user_id)
        print("Known item ID: ", known_item_id)
        X_pred = add_rows_to_csr_matrix(
            X_pred, intended_shape[0] - known_user_id
        )
        print("X_pred after adding rows: ", X_pred.toarray())
        logger.debug(
            f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with items"
        )
        to_predict = predict_frame.value_counts("uid")
        print("To predict: ", to_predict)

        if self.pad_with_popularity:
            popular_items = self.get_popularity_scores(X)
            print("Popular items: ", popular_items)
            for user_id in to_predict.index:
                if user_id >= known_user_id:
                    X_pred[user_id, :] = popular_items
        else:
            row = []
            col = []
            for user_id in to_predict.index:
                if user_id >= known_user_id:
                    row += [user_id] * to_predict[user_id]
                    col += self.rand_gen.integers(0, known_item_id, to_predict[user_id]).tolist()
            pad = csr_matrix((np.ones(len(row)), (row, col)), shape=intended_shape)
            print("Pad: ", pad.toarray())
            X_pred += pad
        print("X_pred after padding: ", X_pred.toarray())
        logger.debug(f"Padding by {self.name} completed")
        return X_pred

    def get_popularity_scores(self, X: csr_matrix):
        """Pad the predictions with popular items for users that are not in the training data."""
        interaction_counts = X.sum(axis=0).A[0]
        sorted_scores = interaction_counts / interaction_counts.max()

        num_items = X.shape[1]
        if num_items < self.K:
            warn("K is larger than the number of items.", UserWarning)

        K = min(self.K, num_items)
        ind = np.argpartition(sorted_scores, -K)[-K:]
        a = np.zeros(X.shape[1])
        a[ind] = sorted_scores[ind]
        
        return a
    # def fit(self, X: InteractionMatrix) -> "Algorithm":
    #     start = time.time()
    #     if not self.historical_data:
    #         self.historical_data = X.copy()
    #     else:
    #         logger.debug(f"Updating historical data for {self.name}")
    #         self.historical_data = self.historical_data + X
    #     end = time.time()
    #     logger.debug(
    #         f"Updated historical data for {self.name} - Took {end - start :.3}s"
    #     )
    #     super().fit(self.historical_data)
    #     return self


class ItemKNNIncrementalMovieLens100K(ItemKNN):
    """Incremental version of ItemKNN algorithm with MovieLens100k Metadata.

    This class extends the ItemKNN algorithm to allow for incremental updates
    to the model. The incremental updates are done by updating the historical
    data with the new data by appending the new data to the historical data.
    """

    def __init__(self, K=10, metadata: pd.DataFrame=None):
        super().__init__(K)
        if metadata is None:
            raise ValueError("Metadata is required for ItemKNNIncrementalMovieLens100K")
        self.metadata = metadata.copy()
        self.training_data: csr_matrix = None

    def append_training_data(self, X: csr_matrix):
        """Append a new interaction matrix to the historical data.

        :param X: The new interaction matrix
        :type X: csr_matrix
        """
        if self.training_data is None:
            return
        X_prev: csr_matrix = self.training_data.copy()
        new_num_rows = max(X_prev.shape[0], X.shape[0])
        new_num_cols = max(X_prev.shape[1], X.shape[1])
        # Pad the previous matrix
        if X_prev.shape[0] < new_num_rows:  # Pad rows
            row_padding = csr_matrix((new_num_rows - X_prev.shape[0], X_prev.shape[1]))
            X_prev = vstack([X_prev, row_padding])
        if X_prev.shape[1] < new_num_cols:  # Pad columns
            col_padding = csr_matrix((X_prev.shape[0], new_num_cols - X_prev.shape[1]))
            X_prev = hstack([X_prev, col_padding])

        # Pad the current matrix
        if X.shape[0] < new_num_rows:  # Pad rows
            row_padding = csr_matrix((new_num_rows - X.shape[0], X.shape[1]))
            X = vstack([X, row_padding])
        if X.shape[1] < new_num_cols:  # Pad columns
            col_padding = csr_matrix((X.shape[0], new_num_cols - X.shape[1]))
            X = hstack([X, col_padding])

        # Merge data
        self.training_data = X_prev + X


    def _fit(self, X: csr_matrix) -> "ItemKNNIncremental":
        """Fit a cosine similarity matrix from item to item."""
        if self.training_data is None:
            self.training_data = X.copy()
        else:
            self.append_training_data(X)
        print("Training data: ", self.training_data.toarray())
        super()._fit(self.training_data)
        return self
    
    def _predict(self, X: csr_matrix, predict_im: InteractionMatrix) -> csr_matrix:
        """Predict the K most similar items for each item using the latest data."""
        X_pred = super()._predict(self.training_data)
        print("In ItemKNNIncremental _predict: ", X_pred.toarray())
        # ID indexing starts at 0, so max_id + 1 is the number of unique IDs
        max_user_id = predict_im.max_user_id + 1
        print("Max user ID: ", max_user_id)
        max_item_id = predict_im.max_item_id + 1
        print("Max item ID: ", max_item_id)
        intended_shape = (
            max(max_user_id, X.shape[0]),
            max(max_item_id, X.shape[1]),
        )
        print("X.shape: ", X.shape)
        print("Intended shape: ", intended_shape)

        predict_frame = predict_im._df
        print("Predict frame: ", predict_frame)

        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = X_pred.shape
        print("Known user ID: ", known_user_id)
        print("Known item ID: ", known_item_id)
        X_pred = add_rows_to_csr_matrix(
            X_pred, intended_shape[0] - known_user_id
        )
        print("X_pred after adding rows: ", X_pred.toarray())
        logger.debug(
            f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with items"
        )
        to_predict = predict_frame.value_counts("uid")
        print("To predict: ", to_predict)

        # pad users with items from most similar user
        user_similarity_matrix = self.get_user_similarity_matrix()
        print("User similarity matrix: ", user_similarity_matrix)
        for user_id in to_predict.index:
            if user_id >= known_user_id:
                most_similar_user_idx = np.argmax(user_similarity_matrix[user_id][:known_user_id])
                X_pred[user_id, :] = X_pred[most_similar_user_idx, :]
    
        print("X_pred after padding: ", X_pred.toarray())
        logger.debug(f"Padding by {self.name} completed")
        return X_pred

    def get_user_similarity_matrix(self):
        user_metadata = self.metadata.copy()

        # set userId as index
        user_metadata.set_index('userId', inplace=True)
        user_metadata.index.name = None

        # reorder the indices
        user_metadata.reset_index(drop=True)
        user_metadata.sort_index(inplace=True)

        # zipcode is a column that does not provide any useful information so we drop it
        user_metadata = user_metadata.drop(columns=['zipcode'])

        # obtain categorical columns
        categorical_columns = user_metadata.select_dtypes(include=['object']).columns.tolist()

        # Use one-hot encoding to encode the categorical columns
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(user_metadata[categorical_columns])

        # obtain the column names for the encoded data
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

        # Concatenate the one-hot encoded dataframe with the original dataframe and drop the original categorical columns
        df_encoded = pd.concat([user_metadata, one_hot_df], axis=1)
        df_encoded = df_encoded.drop(categorical_columns, axis=1)

        # compute cosine similarity but exclude self-similarity
        user_similarity_matrix = cosine_similarity(df_encoded)
        np.fill_diagonal(user_similarity_matrix, 0)

        return user_similarity_matrix

