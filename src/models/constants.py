from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

random_seed = 1
best_model_file_name = 'final_model.pkl'

baseline_classifiers = {
    "LogisiticRegression": LogisticRegression(random_state=random_seed, max_iter=200),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(random_state=random_seed),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_seed),
    "RandomForestClassifier": RandomForestClassifier(random_state=random_seed)
}

LogisiticRegression_grid = {
    "classification__penalty": ['l2'],
    "classification__C": [0.001, 0.01, 0.1, 1],
    "classification__solver": ['lbfgs']
}

model_metrics = {
    'AUC':'roc_auc',
    'RECALL':'recall',
    'PRECISION':'precision',
    'F1':'f1'
}
