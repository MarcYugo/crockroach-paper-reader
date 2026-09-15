<!-- 第 1 页 -->
AutoAlpha: an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment

Tianping Zhang, Yuanqi Li, Yifei Jin, Jian Li
Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University, China
ztp18@mails.tsinghua.edu.cn, {timezerolyq, yfjin1990}@gmail.com,

Abstract

The multi-factor model is a widely used model in quantitative investment. The success of a multifactor model is largely determined by the effectiveness of the alpha factors used in the model. This paper proposes a new evolutionary algorithm called AutoAlpha to automatically generate effective formulaic alphas from massive stock datasets. Specifically, first we discover an inherent pattern of the formulaic alphas and propose a hierarchical structure to quickly locate the promising part of space for search. Then we propose a new Quality Diversity search based on the Principal Component Analysis (PCA-QD) to guide the search away from the well-explored space for more desirable results. Next, we utilize the warm start method and the replacement method to prevent the premature convergence problem. Based on the formulaic alphas we discover, we propose an ensemble learning-to-rank model for generating the portfolio. The backtests in the Chinese stock market and the comparisons with several baselines further demonstrate the effectiveness of AutoAlpha in mining formulaic alphas for quantitative trading.

1 Introduction

Predicting the future returns of stocks is one of the most challenging tasks in quantitative trading. Stock prices are affected by many factors such as company performances, investors’ sentiment, and new government policies, etc. To explain the fluctuation of stock markets, economists have established several theoretical models. Among the most prominent ones, the Capital Asset Pricing Model (CAPM) [Sharpe, 1964] dictates that the expected return of a financial asset is essentially determined by one factor, that is the market excess return, while the Arbitrage Pricing Theory (APT) [Ross, 2013] models the return by a linear combination of different risk factors. Since then, several multi-factor models have been proposed and numerous such factors (also called abnormal returns) have been found in the economics and finance literature. For example, the celebrated Fama-French Three Factor Model [Fama and French, 1993] discovered three important factors that can explain almost 90% of the stock returns¹. In quantitative trading practice, designing novel factors that can explain and predict future asset returns are of vital importance to the profitability of a strategy. Such factors are usually called alpha factors, or alphas in short.

In 2016, the quantitative investment management firm, WorldQuant, made public 101 formulaic alpha factors in [Kakushadze, 2016]. Since then, many quantitative trading methods have used these formulaic alphas for stock trend prediction [Chen et al., 2019]. A formulaic alpha, as the name suggests, is a kind of alpha that can be presented as a formula or a mathematical expression.

Alpha#101 = (close – open)/(high – low)

For example, the above formulaic alpha is one of the alpha factors from [Kakushadze, 2016] and is calculated using the open price, the close price, the highest price and the lowest price of stocks on each trading day. This alpha formula reflects the momentum effect that has been observed in different market (see e.g., [Jegadeesh and Titman, 1993]). For each day, the alpha gives different values for different stocks. The higher the value, it is more likely that the stock will have relatively larger returns in the following days.

There are much more complicated formulaic alphas than the one shown above. In Figure 2, we show two examples, Alpha#71 and Alpha#72 in [Kakushadze, 2016]. The most common way of producing new formulaic alphas is to have economists or financial engineers to come up with new economical ideas, transform these ideas into formulas and then validate its effectiveness on the historical stock datasets. It is known that WorldQuant has been employing a large number of financial engineers and data miners (even part-time online users²) to design new alphas. This way of finding good alphas requires tremendous human labor and expertise, which is not realistic for small firms or individual investors. Therefore, there is an urgent need to develop tools for mining new effective alphas from massive stock datasets automatically.

Our goal is to find as many diverse formulaic alphas with desirable performance as possible within limited computational resources. Unlike many optimization and search problems which aim at finding one single desirable solution, we

¹https://

<!-- 第 2 页 -->
operators
factors
open close high
low wrap volumes

AutoAlpha
AutoAlpha
Stock Ranking
Prediction

Figure 1: The framework of our approach.

Alpha#71: maxT5, Rank(decay, linear(correlation(T5, Rank(close, 3,43976), T5, Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), T5, Rank(decay, linear(rank((low + open) - (vwap + vwap))))*2), 16.4662), 4.4388))

Alpha#72: (rank(decay, linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519))) / rank(decay, linear(correlation(T5, Rank(vwap, 3.72469), T5, Rank(volume, 18.5188), 6.86671), 2.95011)))

Figure 2: Formulas Alpha#71 and Alpha#72.

prefer to look for multiple diverse solutions with high performance and low correlation.

As shown in Figure 3, a formula alpha can be expressed as a tree where the leaves correspond to raw data and the inner nodes correspond to various operators. As the discrete search space is very large, it is natural to use genetic algorithms to search for effective alphas in the form of trees. However, as we argue below, this is not straightforward and there are several challenges we need to address.

Challenge 1: Quickly locate the promising search space. The vanilla genetic algorithm is generally inefficient in mining effective formulaic alphas, due to the fact that effective alphas are sparse in the huge search space. Therefore, how to quickly locate the promising space for search becomes a critical issue.

Challenge 2: Guide the search away from the explored search space. In order to find many diverse and effective formulaic alphas, we need to run the genetic algorithm several times for more results. However, the vanilla genetic algorithm usually converges to the same local minima.

Challenge 3: Prevent the premature convergence problem. The premature convergence problem [Gupta and Ghafir, 2012] arises in genetic algorithms when some type of effective genes dominate the whole population and destroy the diversity in the population. When premature convergence happens, the population stucks at a suboptimal state and we can no longer produce offspring with higher performance.

In this paper, we propose a new model called AutoAlpha to address the above challenges in a unified framework. The technical contributions of this paper can be summarized as follows:

- We discover an inherent pattern of the formulaic alphas. Based on this, we design a hierarchical structure to quickly locate the promising space for search, which address Challenge 1.
- For Challenge 2, we propose a new Quality Diversity method based on the Principal Component Analysis (PCA) to guide the search away from the explored space for more diverse and effective formulaic alphas.
- We introduce the warm start method at the initialization step and the replacement method during reproduction to prevent the premature convergence problem. This addresses Challenge 3.

Based on the formulaic alphas we discover, we propose an ensemble learning-to-rank model to predict the stocks rankings and develop effective stock trading strategies. We perform backtests in the Chinese stock market for different holding periods. The backtesting results show that our method consistently outperforms several baselines and the market index.

2 Problem Statement

Mining formulaic alphas can be regarded as a feature extraction problem. We start from an initial set of basic factors (e.g. open, close, volume, etc.) and operators (e.g. +-s/, min, std, etc.), and then build formulaic alphas that satisfy certain performance measurement criterion, in order to reveal some inherent patterns of the stock market. The basic factors and the operators we use can be found in [Kakushadze, 2016]. The data is public in Chinese stock markets and can be accessed through multiple resources³. In this section, we formalize the problem of mining formulaic alphas.

2.1 Stock Returns

The return of a stock is generally determined by the close price of the stock and the holding period. For a given stock s, a given date t and a given holding period h, the return of the stock can be calculated as:

\[ r_{t,s}^{(h)} = \frac{close_{t+h,s} - close_{t

<!-- 第 3 页 -->
Figure 3: The demonstration of crossover. The leftmost tree shows the tree representation of the formulaic alpha 'close – open)/(high – low)'. The trees on the right are two children after crossover. op: operator. f: factor.

The IC of an alpha indicates the relevance between the alpha and the stock returns, and should be as high as possible⁴.

2.3 Similarity between Alphas

The similarity between the alpha \( i \) and \( j \) is calculated as:

\[ sim(i, j) = \frac{1}{T} \sum_{t=1}^{T} corr(\alpha_t^{(i)}, \alpha_t^{(j)}) \]

A group of alphas is diverse if the similarity between any two alphas in the group is lower than 0.7.

In the process of mining formulaic alphas, our goal is to find as many diverse formulaic alphas with high IC as possible within limited computational resources.

3 AutoAlpha

AutoAlpha is a framework based on genetic algorithms [Whitley, 1994]. Genetic algorithm is a kind of metaheuristic optimization algorithm which draws inspiration from biological process that produces new offspring and evolves a species. The vanilla genetic algorithm uses mechanisms such as reproduction, crossover, mutation and selection to give birth to new offspring. In each step of regeneration, it uses fitness function to select the best-fit individuals for reproduction. After we give birth to new offspring through crossover and mutation operations, we replace the least-fit individuals in the population with new individuals to realize the mechanism of elimination through competition.

In order to apply the genetic algorithm for mining formulaic alphas, first we need to define the genetic representation of a formulaic alpha. As shown in the leftmost tree of Figure 3, a formulaic alpha can be represented as a formulaic tree. It would be much easier for us to carry out crossover and mutation for trees. Figure 3 shows the crossover between two formulaic alphas of depth 2. We perform the crossover in the same depth level to prevent the depth from increasing. That is, the crossover between gene1 and gene3 in Figure 3 is not allowed. The gene2 and gene3 are called root genes which are directly attached to their root operators while gene1 is not.

3.1 Hierarchical Structure

The search space of trees is huge and the effective alphas are very sparse. In our experiment, we find out that the standard genetic algorithm (e.g., that implemented in python package

Figure 4: The IC of the root genes of the top 100 discovered formulaic alphas and its distribution. For example, in the left figure, the blue histogram is the density plot of IC of the formulaic alphas of depth 2 (estimated by 20000 randomly generated samples). And the red histogram is the frequency plot of IC of the root genes of the top 100 discovered formulaic alphas of depth 3.

'gplearn' is generally inefficient in initializing the population and exploring the search space for mining formulaic alphas (see the results in Section 4.4). For remedy, We propose a novel hierarchical search strategy for the genetic algorithm, that is significantly more efficient in the initialization and exploration of the search space.

Motivation

In the early stage of this research, we have been using vanilla genetic algorithms for mining formulaic alphas. An interesting phenomenon occurs during the experiments that the algorithm usually converges to similar formulaic alphas with 'vwap/close' as a piece of its genes, 'vwap' is the Volume Weighted Average Price of a stock. The gene vwap/close itself is also an effective alpha which relates to the phenomenon of mean reversion. While vwap/close itself is an effective formulaic alpha, the formulaic alphas of higher depth which contain vwap/close as a piece of its genes usually combine this mean reversion information with some other information and have higher effectiveness.

Based on such phenomenon, we propose a hypothesis about the inherent pattern of the formulaic alphas, that is, most of the effective alphas have at least one effective root gene. Intuitively, if we want to obtain the formulaic alphas of higher depth, we should search nearby the effective alphas of lower depth. We design an experiment to verifies the hypothesis. First, we use the vanilla genetic algorithm for evolving formulaic alphas. Then we select the top 100 discovered formulaic alphas. For each selected alpha, we further collect its root gene with highest IC. We use the density plot to show that those root genes are effective and are hard to obtain by random generation. The results are shown in the Figure 4.

Based on the analysis, if we maintain a population with diverse