import streamlit as st
import streamlit.components.v1 as components

# ปิด padding ของหน้า (ดูเต็มจอมากขึ้น)
st.set_page_config(layout="wide")

# ฝัง iframe แบบเต็มจอ (100% width, height ใหญ่)
notion_embed_url = "https://infrequent-eagle-7da.notion.site/ebd/246baceec532804eabb1d31e9c9032e4"

# ใช้ iframe เต็มหน้าจอ
components.iframe(notion_embed_url, width=1920, height=1080, scrolling=True)
