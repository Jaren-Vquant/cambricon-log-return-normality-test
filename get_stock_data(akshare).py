import akshare as ak
import pandas as pd
import os
import time 
from datetime import datetime
from dateutil.relativedelta import relativedelta


#获取最新日期往前5年的股票数据
def get_stock_daily_data(symbol):
    try:
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=5)
        end_date = end_date.strftime("%Y%m%d")
        start_date = start_date.strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol,
                                period="daily",
                                start_date=start_date,
                                end_date=end_date,
                                adjust="qfq"
                                )
        
    except Exception as e:
        print(f"错误类型是{e}")
        return None
if __name__ == "__main__":
    #定义股票，这里以寒武纪A股"688256"为例子
    stock_code = "688256"#寒武纪股票代码
 
    #创建存储目录（默认保存在D盘中）
    data_dir = "寒武纪股票数据"
    os.makedirs(data_dir,exist_ok=True)

    #生成保存路径
    file_name = "寒武纪.csv"
    file_path = os.path.join(data_dir,file_name)

    #执行函数获取信息
    df = get_stock_daily_data(stock_code)

    #保存为csv格式
    if df is not None and not df.empty:
        df.to_csv(file_path,index=False,encoding="utf-8-sig")
        print(f"✅已经保存成功！已经保存{len(df)}条记录至{file_path}")
    else:
        print("❌未能保存成功！")


    

