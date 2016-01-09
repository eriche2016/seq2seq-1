# seq2seq
Sequence to Sequence Learning

## Introduction
I recently implemented a deep learning based conversation engine, Musio, which is originally built on top of sequence to sequence learning model. When I saw the paper, A Neural Conversational Model, introduced from Google, I was really inspired. It seemed like generating a relevant sequence given a sequence. Most of existing chatbots used to take full advantage of AIML-like pattern matching or rules to interact with human beings. But the neural conversational model can learn and understand human languages using a large conversational datasets without any rules and complicated feature engineering.

I crawled some subtitles for kids, and developed a sequence to sequence learning algorithm in Torch7. Much of the base code is from lstm-char-cnn(YoonKim) and char-rnn(Karpathy). But I restructured the whole source codes to enable sequence to sequence learning, and embedded encoder-decoder architecture to make it flexible with other algorithms.

Here is the demo I have worked on. The size of dialogue is somewhat small (206,194 utterances), and the time to learn is short (just for 2 days). I used two-layered LSTM RNN model with some options such as the size of word vector(1,000), the size of rnn(2,000), the number of batch(30), and the number of time steps(35). You will be able to see the progress of source codes in my github soon or later.


## How to install
### Dependencies
#### Web Service
```
luarocks install https://raw.githubusercontent.com/benglard/htmlua/master/htmlua-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/benglard/waffle/master/waffle-scm-1.rockspec
luarocks install json
```

## How to run
th generator.lua -model <model>

## Author
Woohyun Kim (deepcoord@gmail.com)
