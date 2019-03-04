#%%
from fastai import *
from fastai.tabular import *

#%%
df = pd.read_csv('titanic/train.csv').drop(['Name'], axis=1)
testdf = pd.read_csv('titanic/processed_test.csv').drop(['Name'], axis=1)
df.head()

#%%
dep_var = 'Survived'
cont_names = ['Fare']
cat_names = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']

#%%
procs = [FillMissing, Categorify, Normalize]
PATH = Path('titanic')


#%%
np.random.seed(41)
test = TabularList.from_df(testdf, path=PATH, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(df, cat_names=cat_names, cont_names=cont_names, procs=procs, path=PATH)
        .random_split_by_pct(0.10)
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch())


#%%
# learn = tabular_learner(data, layers=[150, 300], metrics=accuracy) -> 86% accuracy
# learn = tabular_learner(data, layers=[200, 300], metrics=accuracy) -> 87% accuracy
# learn = tabular_learner(data, layers=[200, 400], metrics=accuracy) -> 87.6% accuracy
# learn = tabular_learner(data, layers=[200, 800], metrics=accuracy) -> 83.1% accuracy
# learn = tabular_learner(data, layers=[200, 600], metrics=accuracy) -> 87.64% accuracy
learn = tabular_learner(data, layers=[200, 400], metrics=accuracy)

#%%
learn.lr_find()
learn.recorder.plot()
#%%
learn.fit_one_cycle(5, 1e-2)

#%%
learn.fit_one_cycle(5, 1e-3)

#%%
# learn.fit_one_cycle(5, 1e-4)

#%%
learn.fit_one_cycle(5, 1e-5)

#%%
# learn.save('model_83_1')


#%%
learn.load('model_83_1')

#%%
preds = learn.

#%%
