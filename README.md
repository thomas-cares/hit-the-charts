# predict a hit

* Modeling the Spotify streaming charts of 20 countries from 2018 to 2020
* evaluating the likelihood of a song meeting the musical qualities of these charts songs
* pass a a Spotify track URL, a Spotify album URL or a list of Spotify track URLs and have an evaluation returned
* set up a charts extraction tool that should allow for feeding the model with an up to date set of charts.

## MOTIVATION
After recording a set of songs choosing the right single to spearhead the promotional campaign is a common challenge faced by artists, labels and their publicists. Picking the right single is decisive for the successfull introduction of a new artist or a new album to the limited attention span of the public. On top it is a major financial commitment as a single release goes along with major expenditures like a music video production or spending on social media ads as well as media plugging form online to radio. 

I'd like to introduce a modes machine learning tool to assist with making the decision for the right song to get things started.
 

## METHODE
I will use supervised learning to model the musical features characterising the songs making up the Spotify streaming charts for the past 3 years in Europe and North America.

### APPLIED MODELS
* random forest
* support vector classifier
* decision tree
* k nearest neighbour

## RUN THE PREDICTOR
The program allows you to pass a list of Spotify song IDs returning an assessment to what extend each song meets the qualities of the songs currently reigning the charts.
* open Jupyter Notebook "charts_predictor.ipynb"
* set your personal values to "my_client_ID" and "my_client_secret": for details see "AUTHENTIFICATION"
* run all cells

## AUTHENTIFICATION
* To run this you have to provide a valid Spotify Client ID and Client Secret
* you have to be registered with [Spotify for developers](https://developer.spotify.com/documentation/general/guides/app-settings/#register-your-app) to obtain those.


## FILES

* "charts_predictor.ipynb" here is where the analysis, the modelling and the predicting is happening

* "fetching-data.ipynb" here is where the scraping is happening. The script actually provides multiple options to adapt the data gathering process. Change the set of countries you want to explore, the time period you want to investigate, the time frequency charts are sampled or if you'd rather look into the Spotify viral charts than the streaming chart.

* the data folder is where the CSVs extracted in the scraping process are dumped. 

* data/regional: The sub folder "regional" is the default target when extracting streaming charts. 

* data/viral is the target when extracting Spotify viral charts. They need to be treated differently as their CSVs are set up and need to be handled differently to turn them into viable pandas data frames. 

* data/kaggle I did use a kaggle dataset with originally 160.000 Spotify songs subsetted for the years of release 2018 to 2020. After excluding all charts songs from the kaggle dataset they were my base of non charting songs to model my charts songs against. 

## DATA SOURCES
* [Spotify Charts](https://www.spotifycharts.de)
* [kaggle spotify dataset](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks)

## CHALLENGES DEALT WITH
* setting up a web scraping tool to extract Spotify charts that can be customised to include the list of countries you need. Also the time period to be extracted as well as the frequency (for ex every day vs every 90 days) can be customised for various investigation purposes.

* cleaning and wrangling the extracted data with regex, pandas and list comprehension

* authentification with Spotify for developers with 

* collecting the music features for thousands of songs through Spotify API, which need some error handling

* exploring, understand and visualisation with maptplotlib, seaborn, pandas

* standardizing continuous variables, labelling categorical variables

* modelling the charts songs against a base of non-charting songs with models such as random forest or the support vector classifier provided by the sklearn library

* setting up a function allowing the user to paste a Spotify song URL to get an evaluation of the respective song in return. 
---
Please note: I am well aware there is a lot more to a song than its musical features to go viral: Lyrics, promotional budgets and the pop cultural phenomenon of memes striking at seemingly random very likely comes into play. In this analysis I merely focus on musical features.

# CONCLUSION
Even though my model achieves decent accuracy scores it is struggling with precision.


## SPOTIFY API FEATURES
Spotify API provides the following features that allowed me to create a model

* acousticness: A confidence measure from 0 to 100 of whether the track is acoustic. 100 represents high confidence the track is acoustic. 
* danceability: describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0 is least danceable and 100 is most danceable. 
* energy: Energy is a measure from 0 to 100 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. 
* instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 100, the greater likelihood the track contains no vocal content. Values above 50 are intended to represent instrumental tracks, but confidence is higher as the value approaches 100. 
* loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. 
* speechiness: detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 100 the attribute value. Values above 66 describe tracks that are probably made entirely of spoken words. Values between 33 and 66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 33 most likely represent music and other non-speech-like tracks. 
* tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
* valence: A measure from 0 to 100 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). 
* popularity: The value will be between 0 and 100, with 100 being the most popular. The popularity is calculated from the popularity of song on spotify. 
* duration: length of the track in seconds. 
* release date: year of which the song's album is released 
