import streamlit as st
import streamlit.components.v1 as components

notion_embed_url = "https://infrequent-eagle-7da.notion.site/ebd/246baceec532804eabb1d31e9c9032e4"

card_html = f"""
<div class="card-wrapper">
  <div class="card">
    <div class="card-header">
      <h3 style="margin:0;">📊 Article: YouTube Data Analysis</h3>
    </div>
    <div class="card-body">
      <iframe
        src="{notion_embed_url}"
        frameborder="0"
        style="width:100%; height:100%; border:0; border-radius: 12px;">
      </iframe>
    </div>
  </div>
</div>

<style>
  html, body {{
    margin: 0;
    padding: 0;
    height: 100%;
  }}
  .card-wrapper {{
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;              
    width: 100vw;               
    padding: 0;
  }}
  .card {{
    width: 95vw;                
    height: 95vh;               
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  .card-header {{
    padding: 14px 20px;
    background: linear-gradient(90deg, #6a11cb, #2575fc); /* gradient ฟ้า-ม่วง */
    color: white;
    font-weight: bold;
    border-bottom: 1px solid rgba(0,0,0,0.06);
    flex: 0 0 auto;
  }}
  .card-body {{
    flex: 1 1 auto;
    overflow: hidden;
  }}
</style>
"""

components.html(card_html, height=1000, scrolling=False)
