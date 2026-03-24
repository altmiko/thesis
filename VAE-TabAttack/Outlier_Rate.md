## **Chi-squared Distribution and Chi-squared Test**



### **Chi-squared Distribution**
- The **Chi-squared distribution** is a **probability distribution** that describes the sum of squared standard normal variables.
- It is primarily used in **statistical inference**, such as hypothesis testing and confidence interval estimation.

#### **Key Properties**
- It is defined by **degrees of freedom (df)**, which determines its shape.
- As **df increases**, the distribution becomes more symmetric.
- It is used in various tests, including **goodness-of-fit, homogeneity, and independence tests**.

#### **Usage for Outlier Detection**
- Used to define a **threshold** for **Mahalanobis distance-based outlier detection**.
- If a point’s Mahalanobis distance squared exceeds a Chi-squared critical value, it is flagged as an outlier.

---

### **Chi-squared Test**
The **Chi-squared test** is a **statistical test** that uses the Chi-squared distribution to evaluate hypotheses about categorical or frequency data.

#### **Types of Chi-squared Tests**
1. **Chi-squared Goodness-of-Fit Test**  
   - Determines whether an **observed categorical distribution** differs from an **expected distribution**.
   - Example: Checking if the observed sales distribution of a product follows the expected market distribution.

2. **Chi-squared Test of Independence**  
   - Tests whether two **categorical variables** are independent.
   - Example: Testing whether gender and purchasing preference are related.

3. **Chi-squared Test for Homogeneity**  
   - Compares distributions across different populations.
   - Example: Checking if two regions have the same distribution of customer satisfaction ratings.

#### **Chi-squared Test Formula**
For a test statistic:
\[
\chi^2 = \sum \frac{(O - E)^2}{E}
\]
where:
- \( O \) = Observed frequency
- \( E \) = Expected frequency

If \( \chi^2 \) exceeds a critical value from the **Chi-squared distribution**, we reject the null hypothesis.

---

**Key Differences**:
| Feature | Chi-squared Distribution | Chi-squared Test |
|---------|----------------------|-------------------|
| **Type** | Probability distribution | Hypothesis test |
| **Used For** | Defining thresholds, confidence intervals, Mahalanobis distance | Goodness-of-fit, independence, homogeneity tests |
| **Applicable Data** | Continuous (e.g., Mahalanobis distance) | Categorical (counts, frequencies) |
| **Formula** | Sum of squared normal variables | \( \chi^2 = \sum \frac{(O - E)^2}{E} \) |

---


## **Mahalanobis Distance and the Chi-squared Distribution**
For a dataset with \( p \) variables (features), the **Mahalanobis distance** of an observation \( X_i \) from the mean \( \mu \) is given by:

\[
D_i = \sqrt{(X_i - \mu)^T \Sigma^{-1} (X_i - \mu)}
\]

where:
- \( X_i \) is a **vector** representing an observation,
- \( \mu \) is the **mean vector** of the dataset,
- \( \Sigma \) is the **covariance matrix**,
- \( \Sigma^{-1} \) is the **inverse covariance matrix**.

### **Chi-squared Distribution Threshold**
If the data follows a multivariate normal distribution, the squared Mahalanobis distance \( D^2_i \) follows a **Chi-squared distribution** with \( p \) degrees of freedom:

\[
D^2_i \sim \chi^2_p
\]

Observations with **large** \( D^2_i \) values (far from the mean) are potential outliers. The threshold for detecting outliers is determined by the **Chi-squared quantile function** at a chosen significance level \( \alpha \):

\[
\text{Threshold} = \sqrt{\chi^2_p(1 - \alpha)}
\]

Commonly used significance levels:
- \( \alpha = 0.05 \) → 95% confidence level
- \( \alpha = 0.01 \) → 99% confidence level

If \( D_i > \sqrt{\chi^2_p(1 - \alpha)} \), the observation is flagged as a potential **outlier**.

### **Outlier Detection with Mahalanobis Distance**

In latent space, we can use the Mahalanobis distance to detect outliers. Given a set of latent vectors \( z_i \) with \(k\) dimensions and their corresponding perturbed vectors \( z_i + \delta_i \), the outlier rate \( \mathcal{O}_r \) can be computed as:

\[
\mathcal{O}_r = \frac{1}{N}\sum_{i=1}^N \mathbb{I}\left(D(z_i + \delta_i) > \sqrt{\chi^2_p(1 - \alpha)}\right).
\]

In input and output spaces, we can compute the outlier rate in a similar manner. Given a set of input vectors \( x_i \) with \(k\) dimensions and their corresponding perturbed vectors \( x_i + \delta_i \), the outlier rate \( \mathcal{O}_r \) can be computed as:

\[
    \mathcal{O}_r = \frac{1}{N}\sum_{i=1}^N \mathbb{I}\left(D(x_i + \delta_i) > \sqrt{\chi^2_p(1 - \alpha)}\right).
\]


## **References**
