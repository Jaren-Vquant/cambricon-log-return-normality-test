import yfinance as yf
import os

def get_stock_daily_data(symbol):
    try:
        df = yf.download(symbol, period="5y", interval="1d")
        return df
    except Exception as e:
        print(f"错误类型是：{e}")
        return None

if __name__ == "__main__":
    stock_code = "688256.SS"  # 寒武纪，上交所加.SS后缀

    data_dir = "寒武纪股票数据"
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "寒武纪.csv")

    df = get_stock_daily_data(stock_code)

    if df is not None and not df.empty:
        df.to_csv(file_path, encoding="utf-8-sig")
        print(f"✅ 已保存 {len(df)} 条记录至 {file_path}")
    else:
        print("❌ 未能保存成功！")