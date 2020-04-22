import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import ( make_pipeline)
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,

)


pio.templates.default = 'plotly_white'

random_seed = 1

baseline_classifiers = {
    "LogisticRegression": LogisticRegression(random_state=random_seed, max_iter=200),
    "KNearest": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_seed)
}

sss = StratifiedKFold(n_splits=5)

processed_df = pd.read_csv('../../data/processed/processed.csv')
X = processed_df.drop('Class', axis=1).values
y = processed_df['Class'].values

mean_fpr = np.linspace(0, 1, 100)

classifier_metrics = {}
for classifier in baseline_classifiers.keys():
    tprs = []
    accs = []
    precision = []
    recall = []
    f1 = []
    auc = []
    for train_idx, test_idx in sss.split(X=X, y=y):

        model = make_pipeline(
            SMOTE(sampling_strategy='minority'),
            baseline_classifiers[classifier]
        )

        model.fit(X[train_idx], y[train_idx])
        baseline_y_pred = model.predict(X[test_idx])
        baseline_y_pred_prob = model.predict_proba(X[test_idx])[:,1]
        fpr, tpr, _ = roc_curve(y[test_idx], baseline_y_pred_prob)
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

        accs.append(model.score(X[test_idx], y[test_idx]))
        precision.append(precision_score(y[test_idx], baseline_y_pred))
        recall.append(recall_score(y[test_idx], baseline_y_pred))
        f1.append(f1_score(y[test_idx], baseline_y_pred))
        auc.append(roc_auc_score(y[test_idx], baseline_y_pred))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    classifier_metrics[classifier] = mean_tpr
    metrics = f'''
    Classifier: {classifier} \n
    accuracy: {np.mean(accs)} \n
    precision: {np.mean(precision)} \n
    recall: {np.mean(recall)} \n
    f1: {np.mean(f1)} \n
    auc: {np.mean(auc)}\n\n
    '''
    print(f"{metrics}")
    f = open(f'../../reports/{classifier}_baseline_result.txt', 'w')
    f.write(metrics)
    f.close()


fig = go.Figure()
baseline_trace = go.Scatter(x=mean_fpr,y=mean_fpr,name='Baseline')
fig.add_trace(baseline_trace)

for classifier in classifier_metrics.keys():
    trace = go.Scatter(
        x=mean_fpr,
        y=classifier_metrics[classifier],
        name=classifier
    )
    fig.add_trace(trace)



fig.layout['hovermode'] = 'x'
fig.update_layout(
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    xaxis_title="FPR",
    yaxis_title="TPR",

)

fig.show()