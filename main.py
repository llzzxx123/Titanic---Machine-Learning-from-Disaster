import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

# 数据加载
Train=os.path.join('c:\\Users\\lenovo\\Desktop','train.csv')
train = pd.read_csv(Train)  # 替换为实际路径
Test=os.path.join('c:\\Users\\lenovo\\Desktop','test.csv')
test = pd.read_csv(Test)

# 安全特征工程函数
def feature_engineering(df):
    _df = df.copy()
    _df['Title'] = _df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    _df['Title'] = _df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr',
                                       'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    _df['Title'] = _df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
    
    _df['FamilySize'] = _df['SibSp'] + _df['Parch'] + 1
    _df['IsAlone'] = (_df['FamilySize'] == 1).astype(int)
    
    _df['Fare'] = pd.cut(_df['Fare'], bins=[-1, 7.91, 14.45, 31.0, 512], labels=[0,1,2,3]).astype(float)
    _df['Age'] = pd.cut(_df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0,1,2,3,4]).astype(float)
    
    _df['HasCabin'] = _df['Cabin'].notna().astype(int)
    
    return _df.drop(['Name','Ticket','Cabin','PassengerId'], axis=1, errors='ignore')

# 数据预处理管道
numeric_features = ['Age', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features),
        ('bool', 'passthrough', ['IsAlone', 'HasCabin'])
    ])

# 模型训练流程
def train_model(X, y):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier()
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        results[name] = np.mean(scores)
        print(f"{name} CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return results

#训练
X = feature_engineering(train)
y = X.pop('Survived')
results = train_model(X, y)

#超参数优化（Optuna）
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return cross_val_score(pipeline, X, y, cv=5, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

#最终模型
best_params = study.best_params
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss'))
])
final_model.fit(X, y)

# 生成预测
test_processed = feature_engineering(test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_model.predict(test_processed)
})
submission.to_csv(os.path.join('c:\\Users\\lenovo\\Desktop','submission.csv'))

print("提交文件已生成")
