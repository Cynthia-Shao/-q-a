import pandas as pd
import re
import matplotlib.pyplot as plt

# ========== 解决matplotlib中文乱码 ==========
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# ========== 【核心修复】读取CSV，兼容答案里自带逗号的情况 ==========
# quotechar='"' 告诉程序：双引号里的所有内容（包括逗号）都属于同一列，不拆分！
# on_bad_lines='skip' 自动跳过解析异常的坏行，不会直接崩程序
df = pd.read_csv(
    "12306_data.csv",
    encoding="utf-8",
    quotechar='"',
    on_bad_lines='skip'
)

# ========== 数据清洗流程 ==========
# 1. 删除空行、空值
df = df.dropna(subset=["原始问题", "原始答案"]).reset_index(drop=True)

# 2. 去除首尾空格、换行符
df["原始问题"] = df["原始问题"].str.strip()
df["原始答案"] = df["原始答案"].str.strip()

# 3. 统一清理文本内多余换行、连续空格
df["原始答案"] = df["原始答案"].str.replace("\n", " ").str.replace("\r", "")
df["原始答案"] = df["原始答案"].apply(lambda x: re.sub(r'\s+', ' ', x))

# 4. 按「问题」去重，删掉重复问答
df = df.drop_duplicates(subset=["原始问题"], keep="first")

# 5. 自动给所有问题做语义分类（可视化用）
def classify_question(q):
    q = str(q).lower()
    if any(k in q for k in ["身份证", "证件", "实名", "身份"]):
        return "证件/实名制"
    elif any(k in q for k in ["购票", "买票", "购买"]):
        return "购票方式"
    elif any(k in q for k in ["退票", "改签", "变更"]):
        return "退票/改签"
    elif any(k in q for k in ["儿童", "学生", "残疾", "优待", "优惠"]):
        return "优惠票"
    elif any(k in q for k in ["12306", "网站", "app", "手机"]):
        return "12306平台"
    elif any(k in q for k in ["发票", "报销", "凭证"]):
        return "发票/报销"
    elif any(k in q for k in ["进站", "检票", "乘车", "站台"]):
        return "乘车/进站"
    elif any(k in q for k in ["行李", "携带", "物品"]):
        return "携带品"
    else:
        return "其他"

df["问题分类"] = df["原始问题"].apply(classify_question)

# ========== 导出清洗完成的标准CSV ==========
# 导出时自动给文本加双引号，从根源解决以后再读取不报错
df.to_csv("12306_cleaned.csv", index=False, encoding="utf-8-sig", quotechar='"')
print("✅ 数据清洗全部完成！已生成干净文件：12306_cleaned.csv")
print(f"📊 最终有效问答总数：{len(df)} 条")
print("\n===== 各分类问答数量统计 =====")
print(df["问题分类"].value_counts())

# ========== 可视化1：问题分类柱状图 ==========
category_counts = df["问题分类"].value_counts()
plt.figure(figsize=(12, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
category_counts.plot(kind="bar", color=colors[:len(category_counts)])

plt.title("12306客服问答-问题分类数量统计", fontsize=15)
plt.xlabel("问题类型", fontsize=12)
plt.ylabel("问答条数", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ========== 可视化2：问题分类占比饼图 ==========
plt.figure(figsize=(9, 9))
plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90
)
plt.title("12306客服问答-各类型占比", fontsize=15)
plt.tight_layout()
plt.show()