# bAbI_task_comparaison
Comparaison between some approaches of the bAbI tasks

The dataset can be found [here](https://research.fb.com/downloads/babi/) (I use the 20 task dataset)
Each task has it particullarities and challenges

We try differents algorithms to solve the baby task, which constits on retrieving different information in some discussion like data.

An example:

![alt tag](https://persagen.com/files/misc/arxiv1502.05698-t1.png)
![alt tag](https://persagen.com/files/misc/arxiv1502.05698-t2.png)
Two approaches are implemented:
- a end to end neuronal newtork, inspired by [this code](https://github.com/1202kbs/MemN2N-Tensorflow)

The architecture can be see below. \\
For further infrmation see [this](https://www.braincreators.com/2018/06/memory-networks/)
![alt tag](http://i.imgur.com/nv89JLc.png)

- a recurrent neural network, inspired by [this code](https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py)

The memn2n and the RNN both have hops, but the memory for memn2n is external while the memory of the RNN is internal.
