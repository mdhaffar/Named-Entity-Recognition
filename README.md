Requirements :

  Speechbrain

  Pytorch >= 1.5.0

  Python >=3.7.0

How to run :

1/ Automatic Speech Recognition

python train_ASR.py wav2vec_ASR.yaml --data_parallel_backend

2/ Text to Embeddings

2.1/ Extract ASR embeddings for the last layer of the ASR

2.2/ Train TTS system (text/extracted embeedings)

Use the SpeechBrain script: (https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech/TTS/tacotron2)

3/ Train a Named entity system

python 

4/ 

Named Entity Recognition
