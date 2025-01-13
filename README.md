# DCL
We use the method of cosine similarity to calculate the contrast loss.

Here we use two methods to calculate, one uses **'einsum'** to calculate the dot product and the paradigm and then calculates the cosine.

Another direct call API uses **'torch.nn.functional.cosine_similarity'** to calculate cosine.

In addition, for positive and negative samples, the first method will have a total of $batchsize*batchsize$ pairs of eigendistances, and we randomly selected $batchsize-1$, and for positive and negative samples, there are $batchsize$ pairs.

For the second method, we calculate the distance of the $batchsize*batchsize$ pair for all feature distances

我们使用余弦相似度计算对比损失的方法。

这里我们使用了两种方法进行计算，一种使用einsum计算点积和范式再计算余弦。

另外一种直接调用API使用torch.nn.functional.cosine_similarity计算余弦。

此外，第一种方法对于正正样本和负负样本，总共会有 $batchsize*batchsize$ 对特征距离，我们随机选取了了 $batchsize-1$ 个，而对于正负样本，就有 $batchsize$ 对。

而对于第二种方法，所有的特征距离我们都计算了 $batchsize*batchsize$ 对的距离


