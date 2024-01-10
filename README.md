# Carrying over algorithm in transformers

The folders contain:

1. **Alpaca 7B:** Contains the code and data to extract the relevant attention patterns from Alpaca 7B (not provided). We also included the arrangement of the data in the residual stream for both after the attention and MLP in each layer and labelled the data according to task and the target digit (first 4 generated tokens only). For the latter labelling we experimented with different labellings, but decided to label the first two generated tokens with the first target digit and use the second target digit for the other generated tokens. This is a bit arbitrary, but does show the model is arranging the data in a particular and interpretable way.
2. **Llemma 7B:** Llemma 7B data. Attention maps of 8 most relevant heads and the residual stream for the first five generated outputs.
3. **Zephyr 7B** Zephyr 7B data. Attention maps of 8 most relevant heads and the residual stream for the first five generated outputs.
4. **Decoder-only:** Includes the data and model used for the generative models for integer addition. The code to generate these is also given, as well as a Jupyter notebook to create the figures in the manuscript.
5. **Encoder-only:** Includes the data and model used for the encoder-only models for integer addition. The code to generate these is also given, as well as a two Jupyter notebooks (one for the learning dynamics and the other for the PCA and attention patterns etc.) to create the figures in the manuscript.
6. **Length generalisation:** Study of generalisation from 3 to 6 digit addition using the encoder only models by padding the input. We consider priming, no priming and finetuning. 
