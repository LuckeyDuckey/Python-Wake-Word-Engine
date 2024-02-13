# Python-Wake-Word-Engine
A python wake word engine made with tensorflow. This repo contains all the scripts needed to create a functioning wake word engine, it collects samples of you saying the wake word, in my case this was used to create a model for the word "Jarvis", it will then also record background audio from you. after that the audio should be preprocessed, during this step the preprocessing will take the sample audios and ofset them to make sure that the wake word is being said in many different places in the audio, it will also do something called spec augment where part of the data is destroyed in some of the audios, this will make the model more robust and reliable. you can also choose to up  sample which will just duplicate some of out data set, you should only do this if you have much larger proportion of background clips than actual samples as if the dataset is biased like that the model will not learn properely you want around 1 sample for every 3 background clips. after all the clips have been preprocessed into MFCC format you can begin training, the model type is a CNN + Dense, once training is complete you will get a graph of the loss over time and the model will be saved with name WWD. Now you can inferenece the model using the inference script, all you have to do for each step i have talked about here is run the script and follow any instractions. beware if you do not provide enough samples or back ground audio that can cause the model to not work well at all, you must also provide quality data it cannot be bad quailty. in same directory as all the scripts you need to create a folder called "Data" this is where the model and preprocessed MFCC's will be saved, in the "Data" folder there should also be a folder called "Bg" and a folder called "Samples" this is where audio clips will be saved.
