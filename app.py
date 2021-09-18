import streamlit as st
import numpy as np
import pandas as pd
import faiss
import time
import altair as alt
import plotly.express as px
from longtext import *
from sentence_transformers import SentenceTransformer
from scipy import stats
import psutil, os, gc, json
from warnings import filterwarnings
#from sklearn.metrics.pairwise import cosine_similarity

filterwarnings('ignore')

# Functions
@st.cache(allow_output_mutation=True)
def load_model(model):
    return SentenceTransformer(model)

def search(query):
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    index.add_with_ids(encoded_data[test.index], np.array(range(0, len(test))))
    query_vector = model.encode([query])
    top_k = index.search(query_vector, 30)
    return [test.iloc[_id] for _id in top_k[1].tolist()[0]], top_k, query_vector

def make_clickable(link, text):
    return f'<a target="_blank" href="{link}">{text}</a>'

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

@st.cache(allow_output_mutation=True)
def load_data(check):
    if check:
        counties = alt.topo_feature('https://raw.githubusercontent.com/Sarvar-Anvarov/streamlit-app/main/data/mo.json', 'collection')
        source = 'https://raw.githubusercontent.com/Sarvar-Anvarov/streamlit-app/main/data/salary.csv'
        data = pd.read_csv('data/data_all.csv')
        map_data = pd.read_csv('data/map.csv')
        full = pd.read_csv('data/full_all.csv')
        encoded_data = pd.read_csv('data/full_all.csv').values.astype('float32', order = 'C')
    
    return counties, source, data, map_data, full, encoded_data


# Data
counties, source, data, map_data, full, encoded_data = load_data(True)
z = abs(stats.zscore(data['salary_from'],nan_policy = 'omit'))
test = data['description']


# Pages
with st.sidebar:
    col1, col2 = st.beta_columns([2.2,6])
    with col1:
        st.write("")
    with col2:
        st.image("data/logo.png")
    page = st.radio('Страница', ['Приветствие','Поиск вакансий','Интересная статистика','Карта'])


# Page 1-Intro
if page=='Приветствие':
    st.markdown(dash, unsafe_allow_html = True)
    st.subheader('Приветствие')
    st.markdown(hello, unsafe_allow_html = True)

# Page 2
if page=='Поиск вакансий':
    st.title('Поиск вакансий')
    st.subheader('Напиши про свои навыки, умения, увлечения и найди работу мечты!')
    
    
    # Sidebar and form
    
    exps=['Нет опыта','От 1 года до 3 лет','От 3 до 6 лет','Более 6 лет']
    with st.sidebar:
        st.text('Фильтр недоступен ввиду малой базы \nданных')
        with st.form(key='form'):
            st.subheader('Фильтр параметров')
            city = st.multiselect('Город(а)', ['Москва', 'Санкт-Петербург'], default='Москва')
            salary = st.slider('Диапазон зарплат, ₽', 0, 150000, (25000, 125000), step=5000)
            exp = st.radio('Опыт работы', exps)
            submit_button = st.form_submit_button(label='Принять')

    # Loading model
    model = load_model('paraphrase-xlm-r-multilingual-v1')
    
    
    # Preparing search function
    text = st.text_area('Только не скромничай!', value=example)
    st.text('Топ вакансий по твоим навыкам (модель в доработке, возможны некорректные вакансии)')
    
    results, top_k, vector = search(str(text))
    res = data.loc[top_k[1][0]]
#    st.write(np.reshape(vector, (1,768)).shape, np.reshape(np.array(full.loc[0]), (1,768)).shape)
    res['similarity'] = [cosine_similarity(np.reshape(vector, (1,768)), np.reshape([full.loc[top_k[1][0]].loc[i]][0][0], (1,768))) for i in full.loc[top_k[1][0]].index]
    
    res['similarity'] = (res['similarity']*100).round(1)
    res['job'] = res.apply(lambda x: make_clickable(x['alternate_url'], x['name']), axis=1)
    res['published_at'] = res['published_at'].apply(lambda x: str(x)[:10])
    res['description'] = res['description'].apply(lambda x: x[:100]) + '...'
    df = (
        res
        [(res['salary_from']>=salary[0]) & (res['area']==city[0])  & (res['experience']==exp)]
        [['job','employer','description','published_at']].reset_index(drop=True).rename(
        columns = {'job':'Вакансия',
                   'employer':'Работодатель',
                   'description':'Описание',
                   'similarity':'Сходство',
                   'published_at':'Дата публикации',
                  }
        )
    ).head(10).reset_index(drop=True).sort_values('Сходство', ascending=False).head(10).reset_index(drop=True)
    df['Сходство'] = df['Сходство'].astype(str) + '%'
                   
    st.write(df.to_html(escape=False), unsafe_allow_html=True)

    
# Page 3    
if page=='Интересная статистика':
    st.title('Интересная статистика')
    
    # Sidebar
    st.sidebar.subheader('Фильтр параметров')
    city = st.sidebar.multiselect('Город(а) для визуализации статистики', ['Москва', 'Санкт-Петербург'], default='Москва')
    
    # Stats
    with st.spinner('Обработка данных'):
 
        ## 1
        fig2 = alt.Chart(
            data[z<3][['experience','salary_from']],
            title='Распределение зарплат по необходимому опыту работы'
        ).mark_boxplot(size=60).encode(
            alt.X('experience', title='Опыт работы', 
                  sort=['Нет опыта', 'От 1 года до 3 лет','От 3 до 6 лет','Более 6 лет']),
            alt.Y('salary_from', title='Зарплата')
        ).configure_axisX(labelAngle=0).configure_title(fontSize=14)

        st.altair_chart(fig2, use_container_width=True)
        
        ## 2
        top15 = data['specializations'].value_counts().head(15).index
        
        fig3 = alt.Chart(
            data[z<3][data['specializations'].isin(top15)][['specializations','salary_from']],
            title='Распределение зарплат по сферам'
        ).mark_boxplot().encode(
            alt.X('salary_from', title='Зарплата'),
            alt.Y('specializations', title='Сфера')
        ).configure_title(fontSize=14)
        
        st.altair_chart(fig3, use_container_width=True)


        ## 3
        fig4 = alt.Chart(
            data[['experience']], 
            title='Количество вакансий по необходимому опыту работы'
        ).mark_bar(size=60).encode(
            alt.X('experience', title='Опыт работы',
                  sort=['Нет опыта', 'От 1 года до 3 лет','От 3 до 6 лет','Более 6 лет']),
            alt.Y('count()', title='Количество вакансий')
        ).configure_axisX(labelAngle=0).configure_title(fontSize=14)

        st.altair_chart(fig4, use_container_width=True)
        
        ## 4
        data['published_at'] = pd.to_datetime(data['published_at'])
        fig6_data = data.groupby([pd.Grouper(key='published_at',freq='H')]).size().reset_index().rename(columns={0:'count'})
        
        fig6 = alt.Chart(
            fig6_data,
            title = 'Динамика публикации вакансий'
        ).mark_line(interpolate='basis').encode(
            alt.X('published_at', title='Дата', axis=alt.Axis(format=("%b %d, %-H:00" ))),
            alt.Y('count', title='Количество вакансий')
        ).configure_title(fontSize=14)
        
        st.altair_chart(fig6, use_container_width=True)
        
        ## 5
        fig5_data = pd.DataFrame(data[['employer']].value_counts().head(15)[::-1]).reset_index().rename(columns={0:'count'})
        
        fig5 = alt.Chart(
            fig5_data,
            title='Топ работодателей по количеству вакансий'
        ).mark_bar().encode(
            alt.X('count', title='Количество вакансий'),
            alt.Y('employer', title='Работодатель', sort='-x')
        ).configure_title(fontSize=14)
        
        st.altair_chart(fig5, use_container_width=True)
        
        
        ## 6
        fig1 = px.pie(
            data_frame = data['schedule'].value_counts().reset_index(),
            names = 'index',
            values = 'schedule',
            labels = {'index':'Тип графика работы',
                      'schedule': 'Количество'},
            title = '<b>Процент типов работ</b>'
        ).update_layout(
            title_x=0.5,
            title_font_size=14
        )

        st.plotly_chart(fig1, use_container_width=True)
        
## Page 4
if page=='Карта':
    st.title('Карта')
    
    # Sidebar
    st.sidebar.subheader('Фильтр параметров')
    city = st.sidebar.selectbox('Город', ['Москва'])
    filt = st.sidebar.selectbox('Параметр', ['Зарплата','Количество вакансий','Вакансии'])
    title = f"Средняя зарплата по округам города {city}" if filt=='Зарплата' else f"Количество вакансий в округах города {city}"
    # Stats
    if filt != 'Вакансии':
        with st.spinner('Обработка данных'):
            multi = alt.selection_multi(fields=['Муниципальный округ', filt], bind='legend')
            map1=alt.Chart(counties).mark_geoshape(
                stroke='white'
            ).encode(
                color=alt.Color(filt+':Q', scale=alt.Scale(scheme="tealblues"), legend=alt.Legend(direction='horizontal',legendX=400, legendY=550, orient='none',gradientLength=200)),
                tooltip=['Муниципальный округ:O', filt+':Q']
            ).transform_lookup(
                lookup='id',
                from_=alt.LookupData(source, 'id', ['Муниципальный округ', filt])
            ).properties(
                width=600,
                height=700
            ).add_selection(
                multi
            ).properties(
                title=title
            ).configure_title(fontSize=14)


            st.altair_chart((map1).configure_view(strokeWidth=0), use_container_width=True)
            
    if filt == 'Вакансии':
        fig = px.scatter_mapbox(map_data, lat="lat", lon="lng", hover_name="name",
                                hover_data=["salary_from", "employment","experience"],
                                color_discrete_sequence=["#636EFA"], zoom=8, height=500, opacity=0.2,
                                title='<b>Расположение всех вакансий на карте</b>')
        fig.update_layout(mapbox_style="light", mapbox_accesstoken=token)
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}).update_layout(title_font_size=14)

        
        st.plotly_chart(fig)
