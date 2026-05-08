import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime

# إعداد واجهة التطبيق
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 نظام تحليل الأسهم الذكي")

# مدخلات المستخدم
ticker = st.sidebar.text_input("أدخل رمز السهم (مثلاً AAPL):", "AAPL")
period = st.sidebar.slider("مدة البيانات التاريخية (بالسنوات):", 1, 5, 2)

if st.button("بدء التحليل الشامل"):
    with st.spinner('جاري جلب البيانات وتحليل الاتجاهات...'):
        # 1. جلب البيانات
        data = yf.download(ticker, start=f"{2024-period}-01-01")
        
        if data.empty:
            st.error("لم يتم العثور على بيانات لهذا السهم.")
        else:
            # 2. التحليل الفني (المتوسطات المتحركة)
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # 3. التنبؤ باستخدام Prophet
            df_prophet = data.reset_index()[['Date', 'Close']]
            df_prophet.columns = ['ds', 'y']
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None) # إزالة المنطقة الزمنية
            
            m = Prophet(daily_seasonality=True)
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=30) # تنبؤ لـ 30 يوم قادم
            forecast = m.predict(future)
            
            # عرض الرسم البياني
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="السعر الحالي"))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="التنبؤ المستقبلي", line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. ملخص التحليل الأساسي (الأخبار)
            st.subheader("📰 آخر أخبار السهم وتحليل التوجه")
            stock_info = yf.Ticker(ticker)
            news = stock_info.news[:5]
            for n in news:
                st.write(f"**{n['title']}**")
                st.caption(f"المصدر: {n['publisher']} | الرابط: {n['link']}")

            st.success("تم التحليل بنجاح!")
