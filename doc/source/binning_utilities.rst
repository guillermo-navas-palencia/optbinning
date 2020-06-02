Utilities
=========


Pre-binning
-----------

.. autoclass:: optbinning.binning.prebinning.PreBinning
   :members:
   :inherited-members:
   :show-inheritance:


Transformations
---------------

The Weight of Evidence :math:`\text{WoE}_i` and event rate :math:`D_i` for each bin are related by means of the functional equations

.. math::

   \begin{align}
   \text{WoE}_i &= \log\left(\frac{1 - D_i}{D_i}\right) + \log\left(\frac{N_T^{E}}{N_T^{NE}}\right) = 
   \log\left(\frac{N_T^{E}}{N_T^{NE}}\right) - \text{logit}(D_i)\\
   D_i &= \left(1 + \frac{N_T^{NE}}{N_T^{E}} e^{\text{WoE}_i}\right)^{-1} = \left(1 + e^{\text{WoE}_i - \log\left(\frac{N_T^{E}}{N_T^{NE}}\right)}\right)^{-1},
   \end{align}

where :math:`D_i` can be characterized as a logistic function of :math:`\text{WoE}_i`, and  :math:`\text{WoE}_i` can be expressed in terms of the logit function of :math:`D_i`. 
The constant term :math:`\log(N_T^{E} / N_T^{NE})` is the log ratio of the total
number of event :math:`N_T^{E}` and the total number of non-events :math:`N_T^{NE}`. This shows that WoE is inversely related to the event rate.

.. autofunction:: optbinning.binning.transformations.transform_event_rate_to_woe

.. autofunction:: optbinning.binning.transformations.transform_woe_to_event_rate


Metrics
-------

Gini coefficient
""""""""""""""""

The Gini coefficient or Accuracy Ratio is a quantitative measure of discriminatory and predictive power given a distribution of events and non-events. The Gini coefficient
ranges from 0 to 1, and is defined by

.. math::

   Gini = 1 - \frac{2 \sum_{i=2}^n \left(N_i^{E} \sum_{j=1}^{i-1} N_j^{NE}\right) + \sum_{k=1}^n N_k^{E} N_k^{NE}}{N_T^{E} N_T^{NE}},

where :math:`N_i^{E}` and :math:`N_i^{NE}` are the number of events and non-events per
bin, respectively, and :math:`N_T^{E}` and :math:`N_T^{NE}` are the total number of
events and non-events, respectively.

.. autofunction:: optbinning.binning.metrics.gini

Divergence measures
"""""""""""""""""""

Given two discrete probability distributions :math:`P` and :math:`Q`. The Shannon entropy
is defined as 

.. math::

   S(P) = - \sum_{i=1}^n p_i \log(p_i).

The Kullback-Leibler divergence, denoted as :math:`D_{KL}(P||Q)`, is given by

.. math::

   D_{KL}(P || Q) = \sum_{i=1}^n p_i \log \left(\frac{p_i}{q_i}\right).

The Jeffrey's divergence or Information Value (IV), is a symmetric measure expressible in terms of the Kullback-Leibler divergence defined by

.. math::

   \begin{align*}
   J(P|| Q) &= D_{KL}(P || Q) + D_{KL}(Q || P) = \sum_{i=1}^n p_i \log \left(\frac{p_i}{q_i}\right) + \sum_{i=1}^n q_i \log \left(\frac{q_i}{p_i}\right)\\ 
   &= \sum_{i=1}^n (p_i - q_i) \log \left(\frac{p_i}{q_i}\right).
   \end{align*}

The Jensen-Shannon divergence is a bounded symmetric measure also expressible in 
terms of the Kullback-Leibler divergence

.. math::

   \begin{equation}
   JSD(P || Q) = \frac{1}{2}\left(D(P || M) + D(Q || M)\right), \quad M = \frac{1}{2}(P + Q),
   \end{equation}

and bounded by :math:`JSD(P||Q) \in [0, \log(2)]`. We note that these measures cannot be directly used whenever :math:`p_i = 0` and/or :math:`q_i = 0`.

.. autofunction:: optbinning.binning.metrics.entropy

.. autofunction:: optbinning.binning.metrics.kullback_leibler

.. autofunction:: optbinning.binning.metrics.jeffrey

.. autofunction:: optbinning.binning.metrics.jensen_shannon

.. autofunction:: optbinning.binning.metrics.jensen_shannon_multivariate