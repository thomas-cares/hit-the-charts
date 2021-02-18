from flask import Flask
app=Flask (__name__)

if __name__ == "__main__":
    app.run()


###################################
@app.route('/index/')
def app():

    # INPUT
    url_album=input().strip(' ')
    predict_hit_album(url_album, continuous_feat,categorical_feat)
    ##############################
    ##############################

    ######
    import pandas as pd
    import numpy as np
    import spotipy
    import spotipy.util as util
    from spotipy.oauth2 import SpotifyClientCredentials
    from spotipy.oauth2 import SpotifyOAuth
    import spotipy.oauth2 as oauth2
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import os
    #visualization option

    #AUTHENTIFICATION
    my_client_id=os.getenv('spoti_CLIENT_ID')
    my_client_secret=os.getenv('spoti_CLIENT_SECRET')
    sp=spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=my_client_id, client_secret=my_client_secret))

    #IMPORT CHARTS CSV - Previously scraped
    top50=pd.read_csv(r'data/analysis/top50_charts_songs_2017-2020.csv')

    # TWEAKING COLUMNS
    top50=top50.rename(columns={'length':'duration_ms'}) #rename length/duration
    top50.date=pd.to_datetime(top50.date, format="%Y-%m-%d", yearfirst=True) #date to datetime
    top50['mode'] = np.where(top50['mode'].apply(str)=='1','Major','Minor')#make mode categorical and name 'Major', 'Minor'
    top50['key'] = 'key_'+top50['key'].apply(str) #make key categorical and label it more readable
    top50['top50']=1#introduce numeric column if Top50 song or not 0/1

    # SUBSETTING UNIQUE TOP 50 TRACKS
    top50_unique=top50.drop(['Position','filename','date','country','Streams'],axis=1).drop_duplicates(ignore_index=True)

    # IMPORT KAGGLE SPOTIFY DATA SET - will be used as baseline
    kaggle=pd.read_csv(r'data/kaggle/data.csv')
    kaggle['group']='base' #labeling kaggle data as "base" so their purpose is clear
    kaggle['top50']=0 #prepare for target
    kaggle['mode'] = np.where(kaggle['mode'].apply(str)=='1','Major','Minor') # make mode categorical and name 'Major', 'Minor'
    kaggle['key'] = 'key_'+kaggle['key'].apply(str) #make key categorical and label it more readable

    # SUBSETTING KAGGLE/BASE DATA for years 2018-2020
    kaggle_year=kaggle[kaggle['year'].isin([2018,2019,2020,2021])].reset_index(drop=True)

    # identify search kaggle data for charts songs and kick them out of base
    base=kaggle_year[-kaggle_year['id'].isin(top50_unique.ID)]

    #CREATING LIST OF FEATURES RELEVANT FOR THIS ANALYSIS
    continuous_feat=[
                "acousticness",
                "danceability",
                "duration_ms",
                "energy",
                "instrumentalness",
                "liveness",
                "loudness",
                "speechiness",
                "tempo",
                "valence"]

    categorical_feat=["key","mode"]

    target_feat=['top50']

    #JOIN CHARTS SONGS AND BASE SONGS
    analysis_true=pd.concat([top50_unique[target_feat+categorical_feat+continuous_feat],
                            base[target_feat+categorical_feat+continuous_feat]],
                        axis=0, ignore_index=True)

    #SCALE AND STANDARDIZE
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(copy=False).fit(analysis_true[continuous_feat]) # copy=False parameter overrites the values in the data frame instead of jus returning the standardized data
    analysis_stan=analysis_true
    analysis_stan[continuous_feat]=pd.DataFrame(StandardScaler(copy=False).fit_transform(analysis_true[continuous_feat]),columns=continuous_feat)

    #CHECKING MULTICOLINIARTIY
    corr = round(analysis_stan.corr(),3)
    f, ax = plt.subplots(figsize=(10, 10))
    sns.set_style("darkgrid")
    mask = np.triu(np.ones_like(corr, dtype=bool)) # removing the other side of the heatmap
    cmap = sns.diverging_palette(230, 20, as_cmap=True) #preparing cmap
    sns.heatmap(corr,mask=mask,cmap=cmap,linewidths=.5,square=True,annot=True)
    plt.show()

    # CREATING DUMMIES OF CATEGORICAL FEATURES
    from sklearn.preprocessing import OneHotEncoder
    #picking only the categroical vairbles : key and mode
    cat = analysis_stan[categorical_feat]
    enc = OneHotEncoder()
    cat_encoded = pd.DataFrame(enc.fit_transform(cat).toarray(), columns = enc.get_feature_names())

    # SETTING X AND y
    X = pd.concat((analysis_stan[continuous_feat], cat_encoded), axis=1)
    y=analysis_stan['top50']

    #SPLIT TEST TRAIN
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4)

    #FUNCTION: FOR EVALUATION
    def get_scores(model,X_test,y_test,prediction):
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        eval_model={
                'Model': model,
                'Score': model.score(X_test, y_test),
                'Precision': precision_score(y_test, prediction, pos_label=1),#knear_pred predicitons of k nearest neighbour
                'Recall': recall_score(y_test, prediction, pos_label=1),
                'F1-Score': f1_score(y_test, prediction, pos_label=1)}
        return pd.DataFrame(eval_model,index=[0])

    # K NEAREST NEIGHBOUR
    from sklearn.neighbors import KNeighborsClassifier
    knear_model = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)
    knear_pred = knear_model.predict(X_test)

    eval_knear_df=get_scores(knear_model,X_test,y_test,knear_pred)

    # DECISION TREE
    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier().fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    eval_dt_df=get_scores(dt_model,X_test,y_test,dt_pred)

    # RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    forest_model = RandomForestClassifier().fit(X_train, y_train)
    forest_pred = forest_model.predict(X_test)

    #evaluate decision tree : eval_forest
    from sklearn.metrics import precision_score, recall_score, f1_score
    eval_forest={
        
                'Model': 'Random Forest',
                'Score': forest_model.score(X_test, y_test),
                'Precision': precision_score(y_test, forest_pred, pos_label=1),
                'Recall': recall_score(y_test, forest_pred, pos_label=1),
                'F1-Score': f1_score(y_test, forest_pred, pos_label=1)}

    eval_forest_df=pd.DataFrame(eval_forest,index=[0])

    # SUPPORT VECTOR MACHINE
    from sklearn.svm import SVC
    svm_model = SVC(probability=True).fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    eval_svm_df=get_scores(svm_model,X_test,y_test,svm_pred)


    ##############################

    # RUN PREDICTOR

    #Function pass album ID
    def predict_hit_album(url_album, continuous_feat,categorical_feat):
        
        # authentification
        my_client_id=os.getenv('spoti_CLIENT_ID')
        my_client_secret=os.getenv('spoti_CLIENT_SECRET')
        sp=spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=my_client_id, client_secret=my_client_secret))
        
        #extract album id
        album_id=re.findall(r'\d\w+',str(url_album))[0]

        #extract ids of each album track
        url_list=[sp.album_tracks(album_id)['items'][i]['id'] for i in range(len(sp.album_tracks(album_id)['items']))]  

        ###
        meta_feat=['ID','Track_Name','Artist','Audio']
        ###
        
        #loop through ids
        list_results=[] # here I ll collect the dictionaries as a list to create a single dataframe 
        for i in url_list:
            

            #set up to call spotify API
            song_id=re.findall(r'\d\w+',str(i))[0]

            meta = sp.track(song_id)
            features = sp.audio_features(song_id)
            analysis= sp.audio_analysis(song_id)

            #create dictionary
            keys=[i for i in meta_feat+continuous_feat+categorical_feat] #these lists should contain the nemes of all features I need
            values=[song_id,
                    meta['name'],#song title
                    meta['artists'][0]['name'],#artist name
                    meta['preview_url'], #audio sinppet 30secs
                    features[0]['acousticness'],
                    features[0]['danceability'],
                    meta['duration_ms'],
                    features[0]['energy'],
                    features[0]['instrumentalness'],
                    features[0]['liveness'],
                    features[0]['loudness'],
                    features[0]['speechiness'],
                    features[0]['tempo'],
                    features[0]['valence'],
                    features[0]['key'],
                    features[0]['mode']]

            df=pd.DataFrame(dict(zip(keys,values)),index=[0])

            #standardize the contnuous feature only: using the scaler i fitted with all continuouse variables
            df[continuous_feat]=pd.DataFrame(scaler.transform(df[continuous_feat]),columns=continuous_feat)

            #recreate dummies for keys
            for i in range(0,12):
                if df['key'][0] == i:
                    df[f'x0_key_{i}']=1.0
                else:
                    df[f'x0_key_{i}']=0.0
            df.drop('key',axis=1,inplace=True)

            #recreate dummies for keys
            df['x1_Major']=np.where(df['mode'] == 1,1.0,0.0)
            df['x1_Minor']=np.where(df['mode'] == 0,1.0,0.0)
            df.drop('mode',axis=1, inplace=True)
            
            #MAKING PREDICTION
            X_NEW=df.drop(['ID','Track_Name','Artist','Audio'],axis=1)
            dt_result=dt_model.predict(X_NEW)
            knear_result=round(knear_model.predict_proba(X_NEW)[0][1],2)
            forest_probability=round(forest_model.predict_proba(X_NEW)[0][1],2) #probablitiy positiv
            svm_probability=round(svm_model.predict_proba(X_NEW)[0][1],2)

            #dictionary
            results_dic={'Artist':df['Artist'][0],
                        'Song':df['Track_Name'][0],
                        #'DecisionTree': np.where(dt_result==1,'approved','failed'),
                        #'K-Near':f'{round(knear_result*100,2)}%',
                        'RandForest':f'{round(forest_probability*100,2)}%',
                        'SVM-Model':f'{round(svm_probability*100,2)}%',
                        'Recommender':round(round(forest_probability*100,0)+round(svm_probability*100,0),0),#addiere forest und SVM                     
                        'Audio':df['Audio'][0],
                        'Actual Popularity': meta['popularity']
                        }
            list_results.append(results_dic)
            
            #RADAR PLOT OF GIVEN SONGS
            labels=np.array(continuous_feat)
            stats=df.loc[0,labels].values

            angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)# close the plot
            stats=np.concatenate((stats,[stats[0]]))
            angles=np.concatenate((angles,[angles[0]]))

            fig=plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, stats, 'o-', linewidth=1, color='white')
            ax.fill(angles, stats, alpha=0.25,color='red')
            ax.set_thetagrids((angles * 180/np.pi)[0:10], labels)
            ax.set_title(f"{meta['artists'][0]['name']} - {meta['name']}")
            ax.grid(True)
            
        result_df=pd.DataFrame(list_results).sort_values(by='Recommender',ascending=False)
        return result_df

    return render_template('index.html',result=result)




