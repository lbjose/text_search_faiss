# text_search_faiss

Implementing faiss for faster and efficient similarity search of dense vectors.

To read more about **faiss**, check out the following<br>

[Gettind Started with faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)<br>
[faiss](https://github.com/facebookresearch/faiss)

Following has been covered here:<br>
1) Creating an faiss index
2) Querying on index
3) Range Quering on index

For creating embeddings I have used Universal Sentence Encoder, for more checkout [here.](https://tfhub.dev/google/universal-sentence-encoder-large/5)

Python version >= 3.6

## Installation
~~~bash
pip install -r requirements.txt
~~~

I have used the **faiss cpu**, for using gpu install
~~~bash
pip install faiss-gpu
~~~

