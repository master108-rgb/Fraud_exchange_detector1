# ১. প্রয়োজনীয় লাইব্রেরি ইম্পোর্ট করা
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ২. কাল্পনিক ডেটা তৈরি করা
# এখানে আমরা কিছু স্বাভাবিক এবং কিছু অস্বাভাবিক (জাল) লেনদেন ডেটা তৈরি করছি।
# বাস্তবে আপনি একটি .csv ফাইল থেকে ডেটা লোড করবেন।

# স্বাভাবিক লেনদেন (Normal transactions)
np.random.seed(42) # একই ফলাফল পাওয়ার জন্য
normal_data = {
    'Transaction_Amount': np.random.normal(50, 20, 1000),  # গড় ৫০, স্ট্যান্ডার্ড ডেভিয়েশন ২০
    'Transaction_Time': np.random.randint(0, 24, 1000),  # দিনের যে কোনো সময়
    'Location_X': np.random.normal(0, 10, 1000), # স্থান X
    'Location_Y': np.random.normal(0, 10, 1000)  # স্থান Y
}
normal_df = pd.DataFrame(normal_data)

# জাল লেনদেন (Fraudulent transactions)
fraud_data = {
    'Transaction_Amount': np.random.normal(500, 100, 5),  # অস্বাভাবিকভাবে বেশি টাকা
    'Transaction_Time': np.random.choice([2, 23], 5), # অস্বাভাবিক সময়ে
    'Location_X': np.random.normal(50, 10, 5), # অস্বাভাবিক স্থানে
    'Location_Y': np.random.normal(50, 10, 5)  # অস্বাভাবিক স্থানে
}
fraud_df = pd.DataFrame(fraud_data)

# ডেটাসেট একসাথে করা
data = pd.concat([normal_df, fraud_df], ignore_index=True)

# ৩. Isolation Forest মডেল তৈরি ও প্রশিক্ষণ (Training)
#contamination: মোট ডেটার মধ্যে কত শতাংশ জাল লেনদেন আছে বলে আমরা অনুমান করি।
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data)

# ৪. নতুন লেনদেন সনাক্ত করা
# এখানে আমরা মডেল ব্যবহার করে ডেটাসেটে কোনটি জাল তা চিহ্নিত করব।
# predict() ফাংশন -1 (জাল) এবং 1 (স্বাভাবিক) রিটার্ন করে।
data['is_fraud'] = model.predict(data)

# ৫. ফলাফল দেখা
# শুধু জাল লেনদেনগুলো ফিল্টার করে দেখানো হচ্ছে
print("সনাক্ত করা জাল লেনদেন:")
fraud_transactions = data[data['is_fraud'] == -1]
print(fraud_transactions)

# ফলাফল আরও ভালোভাবে বোঝার জন্য:
print("\nমোট ডেটাসেট:", len(data))
print("সনাক্ত করা জাল লেনদেন:", len(fraud_transactions))
print("স্বাভাবিক লেনদেন:", len(data[data['is_fraud'] == 1]))