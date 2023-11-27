Requirements :

  Speechbrain

  Pytorch >= 1.5.0

  Python >=3.7.0

How to run :

1/ Automatic Speech Recognition

python train_ASR.py wav2vec_ASR.yaml --data_parallel_backend

2/ Text to Embeddings

2.1/ Extract embeddings for the last layer of the ASR

2.2/ Train Text-to-ASR-Embeddings model (text/extracted embeedings)

Use the SpeechBrain script: (https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech/TTS/tacotron2)

3/ Train a Named entity (NLU) system

python train_NLU_from_TTS_emb.py NER.yaml --data_parallel_backend

4/ Plug the end-to-end ASR and the SLU submodule
