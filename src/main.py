import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, K2
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def discrete(df): 
    df['age'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 70, 100], labels=['child', 'young adult', 'middle aged', 'old', 'very old'])
    df['sex'] = np.where(df['sex'] == 1, 'M', 'F')
    df['trestbps'] = pd.cut(df['trestbps'], bins = [0, 120, 130, 140, 180, float('inf')], labels = ['normal', 'high', 'stage 1', 'stage 2', 'crisis'])
    df['chol'] = pd.cut(df['chol'], bins=[0, 200, 240, float('inf')], labels=['normal', 'high', 'extreme'])

    return df 

def generate_dfs(df): 
    df.rename(columns={"target": "heart_disease"}, inplace=True)
    df = discrete(df)
    columns_needed = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'heart_disease']
    df = df[columns_needed]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=5)

    return (train_df, test_df)

def predict(model, test_df, features): 
    infer = VariableElimination(model)
    predictions = []
    actual = []

    for _, row in test_df.iterrows(): 
        evidence = {feat: row[feat] for feat in features}
        result = infer.query(variables=['heart_disease'], evidence=evidence, show_progress=False)
        pred = result.values.argmax()

        predictions.append(pred)
        actual.append(int(row['heart_disease']))

    return actual, predictions

def evaluate(actual, preds): 
    accuracy = accuracy_score(actual, preds)
    precision = precision_score(actual, preds)
    recall = recall_score(actual, preds)
    f1 = f1_score(actual, preds)
    cm = confusion_matrix(actual, preds)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')
    print('\nConfusion matrix: ')
    print(cm)

def main(): 
    df = pd.read_csv("data/heart.csv")
    train_df, test_df = generate_dfs(df)
    
    # Manual feature fitting
    manual_model = DiscreteBayesianNetwork([
        ('age', 'heart_disease'),
        ('sex', 'heart_disease'),
        ('trestbps', 'heart_disease'),
        ('chol', 'heart_disease'),
        ('fbs', 'heart_disease'),
        ('restecg', 'heart_disease'),
        ('age', 'trestbps'), 
        ('age', 'chol'), 
        ('sex', 'chol'), 
        ('fbs', 'chol'), 
        ('age', 'restecg'), 
        ('trestbps', 'restecg')
    ])
    manual_model.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size = 50)
    features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg']
    actual, preds = predict(manual_model, test_df, features)
    print('Manual model evaluation: ')
    evaluate(actual, preds)

     
    # Structured learning
    hc = HillClimbSearch(train_df)
    best_model_structure = hc.estimate(scoring_method=K2(train_df))
    learned_model = DiscreteBayesianNetwork(best_model_structure.edges())
    learned_model.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size = 50)
    learned_features = [cpd.variable for cpd in learned_model.get_cpds() if cpd.variable != 'heart_disease']
    actual, preds = predict(learned_model, test_df, learned_features)
    print("Learned model evaluation: ")
    evaluate(actual, preds)


    print("Test branch testing")
    
if __name__ == '__main__':
    main() 
