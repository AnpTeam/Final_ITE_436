import streamlit as st

#--- Page Setup ---
about_page = st.Page(
    page="pages/Introduction.py",
    title="Introduction",
    icon=":material/account_circle:",
    default=True,
)

project_1_page = st.Page(
    page="pages/youtube_article.py",
    title="Article YouTube Data Analysis",
    icon=":material/youtube_activity:",
)

project_2_page = st.Page(
    page="pages/MiniProject.py",
    title="Miniproject Classification",
    icon=":material/arrow_circle_right:",
)

project_3_page = st.Page(
    page="pages/CatVsDog.py",
    title="Final ML Project",
    icon=":material/arrow_circle_right:",
)

#Nav setup
pg = st.navigation(
    {
        "Big Data [Project]":[about_page],
        "Projects":[project_1_page,project_2_page,project_3_page],
    }
)

#Share All page
st.sidebar.text("☆*: .｡. o(≧▽≦)o .｡.:*☆")
#pg = st.navigation(pages=[about_page,project_1_page,project_2_page,project_3_page])

#Run Nev
pg.run()