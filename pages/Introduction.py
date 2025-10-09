import streamlit as st
import os


image_path = os.path.join("resource", "picture", "profile.jpg")

st.markdown("""
<style>
h1, h2, h3 {
    color: #3498db;
}

div[data-testid="stImage"] > img {
    border-radius: 50%;
    border: 5px solid #3498db;
    box-shadow: 0px 0px 20px rgba(52, 152, 219, 0.6);
    transition: transform 0.3s ease-in-out;
}
div[data-testid="stImage"] > img:hover {
    transform: scale(1.05);
}

.card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    height: 100%;
}
.card:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}
.card h3 {
    margin-top: 0;
    color: #2c3e50;
}
.card p {
    color: #34495e;
    line-height: 1.6;
}

.skill-badge {
    display: inline-block;
    padding: 6px 12px;
    font-size: 14px;
    font-weight: bold;
    background-color: #3498db;
    color: white;
    border-radius: 15px;
    margin: 4px;
    transition: transform 0.2s;
}
.skill-badge:hover {
    transform: scale(1.1);
    background-color: #2980b9;
}
</style>
""", unsafe_allow_html=True)

# =========================
# ✅ HERO SECTION
# =========================
col1, col2 = st.columns([0.4, 0.6], gap="medium")

with col1:
    if os.path.exists(image_path):
        st.image(image_path, width=250)
    else:
        st.warning(f"⚠️ ไม่พบรูปภาพที่ `{image_path}`")

with col2:
    st.title("Anupap Puangwalaisin", anchor=False)
    st.markdown("👨‍🎓 **Student Code :** `2213111335`")
    st.markdown("🎓 **Faculty :** `เทคโนโลยีสารสนเทศ (Information Technology)`")
    st.markdown("""
    *A skilled Backend Developer with experience in developing mobile application, managing databases, Proficient in programming languages, e.g., Python, Java and experienced in RPA automation using Power Automate. *
    """)

st.divider()

# =========================
# ✅ PROJECTS & SKILLS
# =========================
proj_col, skill_col = st.columns([0.55, 0.45], gap="large")

# --- Skills
with skill_col:
    st.header("🛠️ Skills & Languages")

    st.subheader("Programming Languages")
    st.markdown("""
        <span class="skill-badge">Python</span>
        <span class="skill-badge">Java</span>
        <span class="skill-badge">CSS</span>
        <span class="skill-badge">PHP</span>
        <span class="skill-badge">C#</span>
        <span class="skill-badge">Javascript</span>
    """, unsafe_allow_html=True)

    st.subheader("Languages")
    st.markdown("""
        <span class="skill-badge">Thai (Native)</span>
        <span class="skill-badge">English (Communicative)</span>
        <span class="skill-badge">Japanese (Communicative)</span>        
    """, unsafe_allow_html=True)


# --- Projects
with proj_col:
    st.header("📂 Recent Project")
    with st.expander("RPA : Lost ‘N Found"):
        st.write("""
        Developed end-to-end automation solutions lost item management system using Power Automate.
        """)

    with st.expander("Application : Easy Collect"):
        st.write("""
        Developed and maintained mobile parcel queue applications for dorm, ensuring high performance and usability.
        """)
        st.write("""
          https://github.com/AnpTeam/Easy_Collect
        """)
    with st.expander("Game Development : Bravely Soul"):
        st.write("""
        Developed and maintained backend infrastructure on offline game.
        """)
        st.write("""
            https://github.com/AnpTeam/Bravely_soul
        """)

st.header("💡 Interested in Data Science / Data Mining")
st.markdown("""
    <div class="card">
        <h3>🔍 Data Pattern</h3>
        <p style="text-align: justify;">
            ผมมีความสนใจเป็นพิเศษในการค้นหา <strong>Pattern</strong> หรือรูปแบบ 
            ที่ซ่อนอยู่ในชุดข้อมูล โดยเฉพาะในกรณีที่ข้อมูลแต่ละชุดดูเหมือนไม่เกี่ยวข้องกันในเชิงโครงสร้าง 
            แต่กลับมีผลลัพธ์ที่คล้ายคลึงกันอย่างมีนัยสำคัญ
        </p>
        <p style="text-align: justify;">
            ความคล้ายคลึงที่เกิดขึ้นนี้ แม้จะไม่ได้มีความเชื่อมโยงกันโดยตรงในเชิงระบบหรือเนื้อหา 
            แต่สามารถสะท้อนถึงแนวโน้มร่วมบางอย่างที่ซ่อนอยู่ภายใต้พื้นผิวของข้อมูล 
            ทำให้สามารถคาดการณ์ผลลัพธ์ในอนาคตได้อย่างแม่นยำยิ่งขึ้น
        </p>
        <p style="text-align: justify;">
            สิ่งนี้ทำให้ผมสนใจในเทคนิคต่าง ๆ ของ <strong>Data Mining</strong> เช่น 
            <em>Clustering</em>
            ซึ่งสามารถเปิดเผยความสัมพันธ์ของข้อมูลที่ไม่ชัดเจนในตอนแรก
            และช่วยให้ผมสามารถสร้างโมเดลที่มีประสิทธิภาพสูงขึ้น
            และนำไปต่อยอดในการวิเคราะห์เชิงลึกได้อย่างมีประสิทธิภาพ
        </p>
    </div>
    """, unsafe_allow_html=True)


st.divider()



