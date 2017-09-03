#2017-08-30
"create python environment"
conda create -n hw1 cycler=0.10.0=py27_0 freetype=2.5.5 libpng=1.6.22 matplotlib=1.5.1=np111py27_0 mkl=11.3.3 nose=1.3.7=py27_1 numpy=1.11.1=py27_0 openssl=1.0.2h pandas=0.18.1=np111py27_0 pip=8.1.2=py27_0 pyparsing=2.1.4=py27_0 pyqt=4.11.4=py27_4 python=2.7.12 python-dateutil=2.5.3=py27_0 pytz=2016.6.1=py27_0 qt=4.8.7 scikit-learn=0.17.1 scipy=0.18.0=np111py27_0 setuptools=25.1.6=py27_0 sip=4.16 six=1.10.0=py27_0 sqlite=3.13.0=0 tk=8.5.18 wheel=0.29.0=py27_0 zlib=1.2.8=3 pip untangle==1.1.0

#activate enviroment
activate hw1
cd CSE-6250-BDH_HW1
nosetests tests/test_statistics.py

#read files
events, mortality = read_csv('C:/Users/User/CSE-6250-BDH_HW1/tests/data/statistics/')
#colnames(df)
df.columns
#rownames(df)
df.index

#Column selection, addition, deletion

#Describe shows a quick statistic summary of your data
df.describe()

#https://pandas.pydata.org/pandas-docs/stable/comparison_with_r.html
R	                                pandas
dim(df)	                            df.shape
head(df)	                        df.head()
slice(df, 1:10)	                    df.iloc[:9]
filter(df, col1 == 1, col2 == 1)	df.query('col1 == 1 & col2 == 1')
df[df$col1 == 1 & df$col2 == 1,]	df[(df.col1 == 1) & (df.col2 == 1)]
select(df, col1, col2)	            df[['col1', 'col2']]
select(df, col1:col3)	            df.loc[:, 'col1':'col3']
select(df, -(col1:col3))	        df.drop(cols_to_drop, axis=1) but see [1]
distinct(select(df, col1))	        df[['col1']].drop_duplicates()
distinct(select(df, col1, col2))	df[['col1', 'col2']].drop_duplicates()
sample_n(df, 10)	                df.sample(n=10)
sample_frac(df, 0.01)	            df.sample(frac=0.01)

#some usefule code
mortality.patient_id

    mortality.iloc[0,0]
    mortality.iloc[0:1,0]
    df1 = events.loc[events.patient_id==19,['patient_id','timestamp']]
    df1 = events.loc[events.patient_id == mortality.iloc[0:1,0], ['patient_id', 'timestamp']]
    events.iloc[:, 0:3]
    events.iloc[0,0]
    df2 = events.loc[events.patient_id == 24581, 'patient_id']
    df3 = events['patient_id'] #yeild a series
    df1.describe()
    df3.count()

"Number of events recorded for a given patient."
    alive_event = events[~events.patient_id.isin(mortality.patient_id)]
    dead_event = events[events.patient_id.isin(mortality.patient_id)]

"Count of unique dates on which a given patient visited the ICU"
# grouby.get_group=======================================================
rand = np.random.RandomState(1)
df = pd.DataFrame({'A': ['foo', 'bar'] * 6,
                   'B': rand.randn(12),
                   'C': rand.randint(0, 3, 12)})
gb = df.groupby(['A'])
df.groupby(['A']).get_group('bar')
gp = (for k in gb: k)
#groupby.value_counts()
df.groupby(['A','C'])['C'].value_counts()

"Count Duration (in number of days) between the rst event and last event for a given patient."
dead_event
dead_encounter = dead_event.groupby(['patient_id','timestamp'])['patient_id'].unique()
dead_encounter_count = dead_encounter.value_counts().min()

for k in dead_record_length: print k['timestamp'].unique()
print 'key=' + str(k)
print gb.get_group(str(k))

dead_record_length = dead_event.groupby('patient_id')['timestamp'].unique()
dead_record_length[19] aggregate
#Convert a list of values to a time series
dead_event.groupby('patient_id')['timestamp'].agg(['min','max'])
pd.Timestamp(dead_record_length.loc[19,'max'])-pd.Timestamp(dead_record_length.loc[19,'min'])
pd.to_datatime(dead_record_length.loc[:,'max'])-pd.Timestamp(dead_record_length.loc[19,'min'])

#2017-09-01
"How to use git in pycharm"
#https://www.youtube.com/watch?v=NhFRpFtiHec

"install packages in batch"
#interminal
pip install -r requirements.txt
#requirements.txt
jupyter
pymdptoolbox
numpy

#2017-9-02
#pip install pymdptoolbox
"numpy.distutils.system_info.NotFoundError: no lapack/blas resources found"
#don't know how to solve it.

#row sum
np.sum(prob[1],axis=1)
#col sum
numpy.sum(matrix,axis=0)
#asign value
p =1.0/6 #must have .

#series mean
np.mean(isBadSide)
pd.Series.mean(isBadSide)

#creat zero array
prob = np.zeros((2, 10, 10))
#creat empty array
prob = np.empty((2, 10, 10)) # Not true zero
np.zeros((1, 6)) is array([[ 0.,  0.,  0.,  0.,  0.,  0.]])
np.zeros((1, 6)) is not array([ 0.,  0.,  0.,  0.,  0.,  0.])
#Convert pandas dataframe to numpy array, preserving index
df=df.values
numpyMatrix = df.as_matrix()

#creat a incremental range 1:N
np.arange(10)+1

# true to false
isBadSide = pd.Series([1,1,1,0,0,0])
#not isBadSide
~isBadSide+2

prob[1] = np.triu(isGoodSide_N) # upper triangle matirx

#convert vector to n X 1 matrix
reshape(-1, 1)