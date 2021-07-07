from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse as args

def pre_process_labels(dataset):
    class_to_id_mapping = {} 
    id_to_class_mapping = {}

    classes = list(dataset["class"].unique())
    for i in range(len(classes)):
        class_to_id_mapping[classes[i]] = i
        id_to_class_mapping[i] = classes[i]
    dataset["class"] = dataset["class"].apply(lambda x:class_to_id_mapping[x])
    return dataset, class_to_id_mapping, id_to_class_mapping

def create_train_test(X, Y, test_size):
    # Splitting the dataset into the Training set and Test set
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = test_size, random_state = 0)
    
    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)

def get_parent_args():
    parent_args = args.ArgumentParser(add_help=False)
    parent_args.add_argument('--model_path', '-m', dest='model_path', nargs=1, help='Specify the Trained Model Path')
    parent_args.add_argument()