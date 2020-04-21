### Team_ZHAW
This repository contains the submissions, by team_zhaw, to various [AutoDL](https://autodl.chalearn.org/) challenges.
All the submissions are in the respective challenge format and can easily be run using the [starting kit](https://github.com/zhengying-liu/autodl_starting_kit_stable).

##### AutoCV
* Based on MobileNetV2
* Uses bloated classifiers for increased sample efficiency
* Temporal processing for videos is achieved using a singular 3D convolution before global pooling

##### AutoNLP
* Uses SVM with many N-grams and TF-IDF
* In each iteration a new SVM with a specific N-gram range starts
* The prediction is composed of predictions of each iteration (voting)
* Preprocessing for Chinese and English is different (Tokenizer, N-gram word vs N-gram char)
* HashVectorizer is used for speed

##### AutoSpeech

##### AutoWSL


##### AutoDL
* Based on baseline 3
* Uses bloated classifiers for increased sample efficiency for the vision tasks

