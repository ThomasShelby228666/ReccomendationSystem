import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr


load_dotenv()

# Загрузка данных о фильмах
movies = pd.read_csv("data/movies_with_emotions.csv")

# Обработка путей к постерам
movies["poster_path"] = movies["poster_path"].fillna("").astype(str).str.strip()
movies["poster_url"] = movies["poster_path"].apply(
    lambda x: f"https://image.tmdb.org/t/p/original{x}" if x else "imgs/not_found.png"
)

# Загрузка и обработка документа с описанием фильмов
raw_documents = TextLoader("data/tagged_overview.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
# Создание векторной базы данных
db_movies = Chroma.from_documents(documents, HuggingFaceEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        genre: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 10,
) -> pd.DataFrame:
    """
    Получение семантических рекомендаций фильмов на основе векторного поиска.
    Args:
        query: Текстовый запрос для поиска похожих фильмов
        category: Жанр для фильтрации (опционально)
        tone: Настроение для сортировки (опционально)
        initial_top_k: Количество первоначальных результатов поиска
        final_top_k: Финальное количество рекомендаций

    Returns:
        DataFrame с рекомендованными фильмами
    """
    # Поиск похожих документов в векторной базе данных и получение информации по imdb_id
    recs = db_movies.similarity_search(query, k=initial_top_k)
    movies_list = [(rec.page_content.strip('"').split()[0]) for rec in recs]
    movies_recs = movies[movies["imdb_id"].isin(movies_list)].head(initial_top_k)

    # Фильтрация по категории (жанру), если указана
    if genre != "All":
        movies_recs = movies_recs[movies_recs["simple_genres"] == genre].head(final_top_k)
    else:
        movies_recs = movies_recs.head(final_top_k)

    # Сортировка по эмоциональному тону, если указан
    if tone == "Happy":
        movies_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        movies_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        movies_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        movies_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        movies_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return movies_recs

def recommend_movies(
        query: str,
        category: str = None,
        tone: str = None,
) -> list[tuple[str, str]]:
    """
    Основная функция для рекомендации фильмов.

    Args:
        query: Описание желаемого фильма
        category: Жанр фильма
        tone: Настроение фильма

    Returns:
        Список кортежей (URL постера, подпись) для отображения в галерее
    """
    # Получение рекомендаций
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    # Формирование результатов
    for _, row in recommendations.iterrows():
        overview = row["overview"]
        # Обрезка описания до 30 слов
        truncated_over_split = overview.split()
        truncated_overview = " ".join(truncated_over_split[:30]) + "..."

        # Создание списка производственных компаний
        authors_split = row["production_companies"].split(",")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["production_companies"]

        # Создание подписи для фильма
        caption = f"{row['title']} by {authors_str}: {truncated_overview}"
        results.append((row["poster_url"], caption))

    return results

# Подготовка вариантов для выпадающих списков
categories = ["All"] + sorted(movies["simple_genres"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Создание веб-интерфейса
with gr.Blocks() as dashboard:
    gr.Markdown("# Movie recommendation system")

    # Строка с элементами ввода
    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a movie:",
                                placeholder = "e.g., A film about friendship")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select a mood:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")

    # Галерея с результатами
    output = gr.Gallery(
        label="Recommended movies",
        columns=[5],
        rows=[2],
        object_fit="contain"
    )

    # Привязка функций к кнопке
    submit_button.click(fn = recommend_movies,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    # Запуск веб-приложения
    dashboard.launch(server_name="127.0.0.1", server_port=7860)