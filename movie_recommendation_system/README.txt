MEMBERS:
- Christian Francesco Russo
- Lorenzo Spagnolo
- Matteo Spinato

DATASET DOWNLOAD LINK: https://www.kaggle.com/code/rounakbanik/movie-recommender-systems/input

PROJECT STRUCTURE:
The code of this project is contained in the folder MovieRecommendationSystem/Src, which contains the following subfolders:
    - datasets:
	    - raw: the code on the GitLab repository does not contain the dataset, therefore you will need to download the MovieLens 100k dataset from the link above and then add all its .csv files in this folder!
	    - processed: this folder contains the processed tabular dataset and the graph dataset generated starting from the raw MovieLens dataset.
    - scripts:
	    - data: tabular_dataset_handler.py, graph_dataset_handler.py.
	    - approaches:
	        - filters: popularity_rankings.py, content_based_filtering.py, collaborative_filtering.py, hybrid_filtering.py.
	        - collaborative_filters: SVD_based_CF.py, GNN_based_CF.py.
	        - models: GNN_regression_model.
	- trained_models: this folder contains the trained models saved, and a subfolder runs which contains the training information obtained by means of TensorBoard.
	- Main notebooks files: main_classical_approaches.ipynb, main_GNN_approaches.ipynb, main_embedding_approach.ipynb.
      Note: the main notebooks provide a full demonstration of how the previous files work.
