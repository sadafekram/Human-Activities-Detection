import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import combinations
import math
import scipy
import seaborn as sns

cm = plt.get_cmap('gist_rainbow')
line_styles=['solid','dashed','dotted']

def generateDF(filedir,colnames,sensors,patients,activities,slices):
    # get the data from files for the selected patients
    # and selected activities
    # concatenate all the slices
    # generate a pandas dataframe with an added column: activity
    x=pd.DataFrame()
    for pat in patients:
        for a in activities:
            subdir='a'+f"{a:02d}"+'/p'+str(pat)+'/'
            for s in slices:
                filename=filedir+subdir+'s'+f"{s:02d}"+'.txt'
                #print(filename)
                x1=pd.read_csv(filename,usecols=sensors,names=colnames)
                x1['activity']=a*np.ones((x1.shape[0],),dtype=int)
                x=pd.concat([x,x1], axis=0, join='outer', ignore_index=True, 
                            keys=None, levels=None, names=None, verify_integrity=False, 
                            sort=False, copy=True)
    return x


filedir='data/'
sensNames=[
        'T_xacc', 'T_yacc', 'T_zacc', 
        'T_xgyro','T_ygyro','T_zgyro',
        'T_xmag', 'T_ymag', 'T_zmag',
        'RA_xacc', 'RA_yacc', 'RA_zacc', 
        'RA_xgyro','RA_ygyro','RA_zgyro',
        'RA_xmag', 'RA_ymag', 'RA_zmag',
        'LA_xacc', 'LA_yacc', 'LA_zacc', 
        'LA_xgyro','LA_ygyro','LA_zgyro',
        'LA_xmag', 'LA_ymag', 'LA_zmag',
        'RL_xacc', 'RL_yacc', 'RL_zacc', 
        'RL_xgyro','RL_ygyro','RL_zgyro',
        'RL_xmag', 'RL_ymag', 'RL_zmag',
        'LL_xacc', 'LL_yacc', 'LL_zacc', 
        'LL_xgyro','LL_ygyro','LL_zgyro',
        'LL_xmag', 'LL_ymag', 'LL_zmag']
actNames=[
    'sitting',  # 1
    'standing', # 2
    'lying on back',# 3
    'lying on right side', # 4
    'ascending stairs' , # 5
    'descending stairs', # 6
    'standing in an elevator still', # 7
    'moving around in an elevator', # 8
    'walking in a parking lot', # 9
    'walking on a treadmill with a speed of 4 km/h in flat', # 10
    'walking on a treadmill with a speed of 4 km/h in 15 deg inclined position', # 11
    'running on a treadmill with a speed of 8 km/h', # 12
    'exercising on a stepper', # 13
    'exercising on a cross trainer', # 14
    'cycling on an exercise bike in horizontal positions', # 15
    'cycling on an exercise bike in vertical positions', # 16
    'rowing', # 17
    'jumping', # 18
    'playing basketball' # 19
    ]
actNamesShort=[
    'sitting',  # 1
    'standing', # 2
    'lying.ba', # 3
    'lying.ri', # 4
    'asc.sta' , # 5
    'desc.sta', # 6
    'stand.elev', # 7
    'mov.elev', # 8
    'walk.park', # 9
    'walk.4.fl', # 10
    'walk.4.15', # 11
    'run.8', # 12
    'exer.step', # 13
    'exer.train', # 14
    'cycl.hor', # 15
    'cycl.ver', # 16
    'rowing', # 17
    'jumping', # 18
    'play.bb' # 19
    ]

ID=301178
s=ID%8+1
patients=[s]  # list of selected patients
activities = list(range(1,20))
Num_activities=len(activities)
NAc=19 # total number of activities
actNamesSub=[actNamesShort[i-1] for i in activities] # short names of the selected activities
sensors=list(range(0,45)) # list of sensors
print(f"Number of Sensors: {len(sensors)}")
sensNamesSub=[sensNames[i] for i in sensors] # names of selected sensors
Nslices=30 # number of slices to plot
train_slices = list(range(1,Nslices+1))# first Nslices to plot
test_slices = list(range(Nslices+1,60+1))
fs=25 # Hz, sampling frequency
samplesPerSlice=fs*5 # samples in each slice

train_data = generateDF(filedir,sensNamesSub,sensors,patients,activities,train_slices)
test_data = generateDF(filedir,sensNamesSub,sensors,patients,activities,test_slices)

sens_types = ['acc','gyro','mag']
titles = ['Accelerometer', 'Gyroscope', 'Magnetometer']
for s in range(len(sens_types)):
    print(s)
    temp_sens = []
    
    for i in sensNames:
        if sens_types[s] in i:
            temp_sens.append(i)

    centroids=np.zeros((NAc,len(temp_sens)))# centroids for all the activities
    stdpoints=np.zeros((NAc,len(temp_sens)))# variance in cluster for each sensor

    for act in range(1,NAc+1):
        x = train_data[train_data['activity']==act].drop(columns=['activity'])[temp_sens]
        #x = x.rolling(125).mean()

        centroids[act-1,:]=x.mean().values
        stdpoints[act-1]=np.sqrt(x.var().values)

    d=np.zeros((NAc,NAc))
    for k in range(NAc):
        for j in range(NAc):
            d[k,j]=np.linalg.norm(centroids[k]-centroids[j])

    dd=d+np.eye(NAc)*1e6# remove zeros on the diagonal (distance of centroid from itself)
    dmin=dd.min(axis=0)# find the minimum distance for each centroid
    dpoints=np.sqrt(np.sum(stdpoints**2,axis=1))
    plt.figure(figsize=(5,3))
    plt.plot(dmin,label='minimum centroid distance', c='g')
    plt.plot(dpoints,label='mean distance from points to centroid', c='r')
    plt.ylim([0,40])
    plt.grid()
    plt.xticks(np.arange(NAc),actNamesShort,rotation=90)
    plt.legend(loc='upper left')
    plt.title(f"Results with only {titles[s]} Measurements")
    plt.tight_layout()
    plt.show()

print('Start Plotting Magnetometers!')

bench_1=[
        'T_xmag', 'T_ymag', 'T_zmag',
        'RA_xmag', 'RA_ymag', 'RA_zmag',
        'LA_xmag', 'LA_ymag', 'LA_zmag',
        'RL_xmag', 'RL_ymag', 'RL_zmag',
        'LL_xmag', 'LL_ymag', 'LL_zmag']

rollings=['yes', 'no']
sns.set_style("dark")

for roll in rollings:
    
    plt.figure(figsize=(5,3))
        
    centroids=np.zeros((NAc,len(bench_1)))# centroids for all the activities
    stdpoints=np.zeros((NAc,len(bench_1)))# variance in cluster for each sensor

    for act in range(1,NAc+1):
        x = train_data[train_data['activity']==act].drop(columns=['activity'])[bench_1]
        if roll=='yes':
            x = x.rolling(50).mean()

        centroids[act-1,:]=x.mean().values
        stdpoints[act-1]=np.sqrt(x.var().values)

    d=np.zeros((NAc,NAc))
    for k in range(NAc):
        for j in range(NAc):
            d[k,j]=np.linalg.norm(centroids[k]-centroids[j])

    dd=d+np.eye(NAc)*1e6# remove zeros on the diagonal (distance of centroid from itself)
    dmin=dd.min(axis=0)# find the minimum distance for each centroid
    dpoints=np.sqrt(np.sum(stdpoints**2,axis=1))
    plt.plot(dmin,label='minimum centroid distance', c='g')
    plt.plot(dpoints,label='mean distance from points to centroid', c='r')
    plt.grid()
    plt.xticks(np.arange(NAc),actNamesShort,rotation=90)
    plt.legend(loc='upper left')
    if roll=='yes':
        plt.title(f"Results with with 50 Sliding Window Size")
    else:
        plt.title(f"Results with with No Rolling Mean")
    plt.tight_layout()
    plt.show()
    
print('Start Finding The Best Set of Sensors Among Magnetometers!')


X_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]
X_test, y_test = test_data.iloc[:,:-1], test_data.iloc[:,-1]

kmeans = KMeans(n_clusters=19, random_state=0)

fixed_sens = bench_1.copy()
lst = list(combinations(bench_1,5))
result = []
for sens in lst:
    
    print('Changed Sensor:', sens)
    fixed_sens = list(sens)

    kmeans = kmeans.fit(X_train[fixed_sens].rolling(25, min_periods=1).mean())
    y_Xpred = kmeans.predict(X_train[fixed_sens].rolling(25, min_periods=1).mean())

    unique_labels = np.unique(y_Xpred) # Get the unique cluster labels
    label_map = {} # Create a dictionary to map cluster labels to original labels

    for label in unique_labels:
        indices = np.where(y_Xpred == label)[0] # Get the indices of all samples that belong to this cluster
        original_labels = y_train.values[indices] # Get the original labels for these samples
        most_common_label = Counter(original_labels).most_common(1)[0][0] # Determine the most common original label for this cluster
        label_map[label] = most_common_label # Map the cluster label to the most common original label

    y_converted = np.array([label_map[pred] for pred in y_Xpred])
    cm = confusion_matrix(y_train, y_converted)
    cm_1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc = cm_1.diagonal()
    acc_before = sum(acc)/len(acc)
    fixed_sens = fixed_sens.copy()
    result.append(acc_before)
    print('ACC: ', acc_before)
    
    
find_sensors = ['RL_zacc', 'RL_xacc', 'RA_xmag', 'T_zmag', 'LA_ymag', 'LA_zmag', 'LL_ymag']
    
X_train = X_train[find_sensors].rolling(125, min_periods=1).mean()
kmeans = KMeans(n_clusters=19, random_state=0, init='k-means++', n_init=5, algorithm='lloyd')
kmeans = kmeans.fit(X_train)
y_Xpred = kmeans.predict(X_train)

unique_labels = np.unique(y_Xpred) # Get the unique cluster labels
label_map = {} # Create a dictionary to map cluster labels to original labels

########################################## Train ######################################


print('Train')

for label in unique_labels:
    indices = np.where(y_Xpred == label)[0] # Get the indices of all samples that belong to this cluster
    original_labels = y_train.values[indices] # Get the original labels for these samples
    most_common_label = Counter(original_labels).most_common(1)[0][0] # Determine the most common original label for this cluster
    label_map[label] = most_common_label # Map the cluster label to the most common original label
    # Now you can use the label_map dictionary to map the cluster labels to the original labels

y_converted_train = np.array([label_map[pred] for pred in y_Xpred])
cm = confusion_matrix(y_train, y_converted_train)
cm_1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
acc = cm_1.diagonal()
acc_before = sum(acc)/len(acc)
print(acc)
print('Acc Train: ', acc_before)

########################################## Test ######################################

print('Test')

X_test = X_test[find_sensors].rolling(125, min_periods=1).mean()

y_Xpred_test = kmeans.predict(X_test)

unique_labels = np.unique(y_Xpred_test) # Get the unique cluster labels
label_map = {} # Create a dictionary to map cluster labels to original labels

for label in unique_labels:
    indices = np.where(y_Xpred_test == label)[0] # Get the indices of all samples that belong to this cluster
    original_labels = y_test.values[indices] # Get the original labels for these samples
    most_common_label = Counter(original_labels).most_common(1)[0][0] # Determine the most common original label for this cluster
    label_map[label] = most_common_label # Map the cluster label to the most common original label
    # Now you can use the label_map dictionary to map the cluster labels to the original labels

y_converted = np.array([label_map[pred] for pred in y_Xpred_test])
cm_test = confusion_matrix(y_test, y_converted)
cm_1 = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
acc = cm_1.diagonal()
acc_before = sum(acc)/len(acc)
print(acc)
print('Acc Test: ', acc_before)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=actNamesShort)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=actNamesShort)
disp_test.plot()
