import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
日对数收益率序列

计算日对数收益率原因:
1. 时间可加性（最重要的优点）(对数把乘法变成加法。ln(A×B) = ln(A) + ln(B)，多天收益就是各天之和)
2. 对称性(对数收益率中"涨ln(1.5)"和"跌ln(1/1.5)"大小完全相等、符号相反，天然对称)
3. 近似等于普通收益率（小幅波动时）(当日涨跌幅很小时，对数收益率 ≈ 普通收益率，误差可以忽略。A股日常涨跌多在 ±3% 以内，对数收益率误差不超过0.05%，可以直接当普通收益率用)
4. 统计性质好（正态分布假设）(假设对数收益率服从正态分布，价格自然服从对数正态分布，永远大于0，不会出现股价小于0的状态导致模型崩盘)
5. 连续复利的自然语言(对数收益率建模用的就是连续复利，所以公式最简洁。在期权定价等数学推导中，P_T = P_0 × e^(rT) 比 (1+r/n)^(nT) 好求导得多)
6. 量纲归一化 / 数值稳定（对数收益率处理的是价格的比值，天然就不受绝对价格影响。即不同价格的股票山上涨了不同的价格但是最后收益率可以保持一样，可以相互比较）

"""
#读取数据和数据的预处理
df = pd.read_csv("...",#此处写获得的股票数据的绝对路径
                 skiprows=1,#跳过第一行非数据行
                 header=None,
                 names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
                )
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

#计算对数收益率序列
df["log_return"] = np.log(df["Close"]/df["Close"].shift(1))
df = df.dropna(subset=["log_return"])
df = df.set_index("Date")
print(df.head(10))
print("-"*50 + "\n")

"""
输出核心统计指标

要求输出均值、标准差、偏度、峰度
最后输出结果是：
均值 Mean:0.002291
标准差 Std:0.043694
偏度 Skewness:0.618704
峰度 Kurtosis:3.794078
均值 0.002291（日均收益率 +0.23%）：每个交易日平均涨 0.23%，是正的，说明这5年整体趋势向上。换算成年化收益率是 0.002291 × 252 ≈ 57.7%，非常高，符合寒武纪作为 AI 芯片龙头这几年的行情
标准差 0.043694（日波动率 4.37%）每天平均波动 4.37%，换算年化波动率是 0.043694 × √252 ≈ 69.4%。这个数字极高，A 股普通股票年化波动率一般在 30-40%，寒武纪接近 70%，说明它是一只极度高波动的股票，涨得猛但跌得也猛
偏度 +0.618704（右偏）大于 0 说明分布向右偏，意味着偶尔会出现较大的单日暴涨。极端上涨比极端下跌更猛烈，符合题材股的特征——利好消息来了容易一字涨停
峰度 3.794078（尖峰厚尾）远大于 0，说明极端行情出现的频率比正态分布预测的更高。也就是说，用正态分布来预测它的风险会低估尾部风险，现实中暴涨暴跌的概率比模型预期的更大。这是金融数据的普遍现象，叫做"肥尾"
总结：寒武纪这5年是一只高收益、高风险、右偏、厚尾的典型题材股，拿住了赚很多，但过程中波动极大，普通投资者很难真正拿住
"""
print("以下是核心统计指标")
print(f"均值 Mean:{df["log_return"].mean():.6f}")
print(f"标准差 Std:{df["log_return"].std():.6f}")
print(f"偏度 Skewness:{df["log_return"].skew():.6f}")
print(f"峰度 Kurtosis:{df["log_return"].kurt():.6f}")#注意:pandas算出来的是超额峰度


"""
绘制分布直方图并叠加曲线

需要用到scipy里的stats模块
需要使用matplotlib.pyplot模块

对于得到的直方图解读:
整体形状:分布中心在0附近，说明大多数交易日涨跌幅很小，接近0，这是正常的。
橙色核密度曲线 vs 红色正态分布曲线:
橙色是真实分布，红色是理论正态分布，两者差距很明显：橙色峰顶比红色高很多，说明收益率集中在0附近的天数比正态分布预测的更多，这就是之前说的峰度 3.79 远大于0的体现，分布更"尖"。
橙色两侧的尾巴比红色更厚，说明极端暴涨暴跌出现的频率比正态分布预测的更高，这就是肥尾。
右偏：
仔细看直方图和橙色曲线，右边（正收益方向）的尾巴比左边更长，0.15到0.20那里还有一根孤立的柱子，说明偶尔会出现极端暴涨。这就是之前偏度 +0.618 大于0的体现。
结论：
红色和橙色的差距告诉你：用正态分布来预测寒武纪的风险是不够准确的，它低估了极端行情发生的概率。真实分布比正态分布更尖、尾巴更厚、略微右偏，这是 A 股题材股的典型特征。
"""
plt.rcParams["font.sans-serif"] = ["SimHei"]#设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False#解决负号显示的问题

fig, ax = plt.subplots(figsize=(10, 6))#创建画布和坐标

mu = df['log_return'].mean()
sigma = df['log_return'].std()
x = np.linspace(df['log_return'].min(), df['log_return'].max(), 200)

ax.hist(df['log_return'], bins=50, density=True, color='steelblue', alpha=0.6, label='频率分布')
ax.plot(x, stats.norm.pdf(x, mu, sigma), color='red', linewidth=2, label='正态分布曲线')
ax.plot(x, stats.gaussian_kde(df['log_return'])(x), color='orange', linewidth=2, label='核密度曲线')

ax.set_title('寒武纪日对数收益率分布')
ax.set_xlabel('日对数收益率')
ax.set_ylabel('频率密度')
ax.legend()
plt.show()

"""
正态性检验:用数学方法来验证"数据是否服从正态分布"，把图里看到的直觉变成一个有统计依据的结论。我们从图里看出来橙色和红色有差距，但这只是"肉眼判断"，不够严谨。正态性检验给你一个具体的数字结论：是或不是正态分布。
使用scipy.stats.normaltest(原理是同时检验两件事：偏度是否显著不为0;峰度是否显著不为0。是用来检验数据是否服从正态分布的函数。)
"""
stat, p_value = stats.normaltest(df['log_return'])
print(f"统计量：{stat:.4f}")
print(f"p值：{p_value:.6f}")

if p_value < 0.05:
    print("结论：拒绝正态分布假设，数据不服从正态分布")
else:
    print("结论：无法拒绝正态分布假设，数据可能服从正态分布")
