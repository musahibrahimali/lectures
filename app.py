import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import helpers as hp
import matplotlib.pyplot as plt


@st.cache(allow_output_mutation=True, persist=True)
def load_data(file_name: str, names=None, engine: str = "openpyxl") -> any:
    if names is None:
        names = hp.column_names
    data = pd.read_excel(file_name, names=names, engine=engine)
    return data


st.set_page_config(
    page_title="KNUST LECTURE ASSESSMENT",
    layout="wide",
    page_icon="./assets/knust_seal.png",
)

# # --- PAGE TITLE AND KNUST SEAL ---
left_, right_ = st.columns([1, 4.5])
with right_:
    st.title("📊  KNUST LECTURE ASSESSMENT")
with left_:
    st.image(
        "./assets/knust_seal.png",
        width=50,
    )
st.markdown("##")

# SIDEBAR FILE UPLOAD
st.sidebar.subheader("Feed me your excel file")
uploaded_file = st.sidebar.file_uploader(
    "Choose an XLSX file",
    type="xlsx",
    accept_multiple_files=False,
)

if uploaded_file:
    dataframe = load_data(file_name=uploaded_file)
    # the query expression
    query_expression = ""

    lectures = hp.get_unique_values(dataframe=dataframe, column="STAFFID")
    courses = hp.get_unique_values(dataframe=dataframe, column="COURSECODE")
    faculties = hp.get_unique_values(dataframe=dataframe, column="FACULTY")
    departments = hp.get_unique_values(dataframe=dataframe, column="DEPARTMENT")
    colleges = hp.get_unique_values(dataframe=dataframe, column="COLLEGE")
    comments = hp.get_unique_values(dataframe=dataframe, column="COMMENT")

    stats_y_values = [
        len(lectures),
        len(faculties),
        len(courses),
        len(departments),
        len(colleges),
        len(comments),
    ]

    stats_df = pd.DataFrame(
        stats_y_values,
        columns=["COUNT"]
    )

    stats_bar_plot = px.bar(
        stats_df,
        x=hp.statistics_column_names,
        y="COUNT",
        color="COUNT",
        title="TOTAL NUMBERS REPRESENTATION",
        template="plotly_white"
    )

    # let user decide to show the raw data or the grouped data
    # or both
    st.sidebar.markdown(
        "Select the options below to show the related data data"
    )
    right_box, left_box = st.sidebar.columns(2)

    with right_box:
        show_general_stats = st.checkbox(
            "General Stats",
        )
        filter_multi_cols_checkbox = st.checkbox(
            "Multi Selectors",
        )
    with left_box:
        show_grouped_data = st.checkbox("Grouped Data")
        show_raw_data = st.checkbox("Raw Data")

    if not filter_multi_cols_checkbox:
        filter_by_column = st.sidebar.selectbox(
            "SELECT COLUMN",
            options=hp.filter_column_names,
        )

        group_by_fields = st.sidebar.selectbox(
            "SELECT FIELD",
            options=hp.get_unique_values(dataframe=dataframe, column=filter_by_column),
        )

        if filter_by_column == "STAFFID":
            query_expression = query_expression + "STAFFID==@group_by_fields &"
        if filter_by_column == "COURSECODE":
            query_expression = query_expression + "COURSECODE==@group_by_fields &"
        if filter_by_column == "FACULTY":
            query_expression = query_expression + "FACULTY==@group_by_fields &"
        if filter_by_column == "DEPARTMENT":
            query_expression = query_expression + "DEPARTMENT==@group_by_fields &"
        if filter_by_column == "COLLEGE":
            query_expression = query_expression + "COLLEGE==@group_by_fields &"

    if filter_multi_cols_checkbox:
        # filter by multiple columns
        filter_columns = st.sidebar.multiselect(
            "SELECT COLUMNS",
            options=hp.filter_column_names,
            default=["STAFFID", "COURSECODE", "FACULTY", "DEPARTMENT", "COLLEGE"],
        )
        college = st.sidebar.selectbox(
            "COLLEGE",
            options=colleges,
        )
        faculty = st.sidebar.selectbox(
            "FACULTY",
            options=faculties,
        )
        department = st.sidebar.selectbox(
            "DEPARTMENT",
            options=departments,
        )
        course_code = st.sidebar.selectbox(
            "COURSE CODE",
            options=courses,
        )
        staff_id = st.sidebar.selectbox(
            "STAFF ID",
            options=lectures,
        )

        for column in filter_columns:
            if column == "STAFFID":
                query_expression += "STAFFID==@staff_id &"
            elif column == "COURSECODE":
                query_expression += "COURSECODE==@course_code &"
            elif column == "FACULTY":
                query_expression += "FACULTY==@faculty & "
            elif column == "DEPARTMENT":
                query_expression += "DEPARTMENT==@department &"
            elif column == "COLLEGE":
                query_expression += "COLLEGE==@college &"

    if query_expression.endswith("&"):
        query_expression = query_expression[:-1].strip()

    # GROUP THE DATA
    grouped_data = dataframe.query(
        query_expression,
    )

    if len(grouped_data) <= 0:
        st.title("NO DATA TO DISPLAY")
        st.markdown("NO DATA FOR THIS SELECTION")
    else:
        # # # --- GET STUFF UP ---
        stats_y_values = [
            len(lectures),
            len(faculties),
            len(courses),
            len(departments),
            len(colleges),
            len(comments),
        ]

        stats_df = pd.DataFrame(
            stats_y_values,
            columns=["COUNT"]
        )

        stats_bar_plot = px.bar(
            stats_df,
            x=hp.statistics_column_names,
            y="COUNT",
            color="COUNT",
            title="TOTAL NUMBERS REPRESENTATION",
            template="plotly_white"
        )

        questions_dataframe = hp.get_question_dataframe(
            data_frame=grouped_data if not show_general_stats else dataframe,
            start="Q1",
            end="Q15"
        )
        main_counts = hp.count_entries(questions_dataframe)
        resp_count_dict = hp.response_counts(main_counts)

        resp_df = pd.DataFrame(
            sorted(resp_count_dict.values(), reverse=True),
            columns=["COUNT"]
        )

        stats_df_panel = pd.DataFrame(
            [stats_y_values],
            columns=hp.statistics_column_names,
        )

        responses_bar_plot = px.bar(
            resp_df,
            x=hp.response_categories,
            y="COUNT",
            color="COUNT",
            title="TOTAL RESPONSE REPRESENTATION",
            template="plotly_white"
        )

        resp_pie_chart = px.pie(
            resp_df,
            values="COUNT",
            names=hp.response_categories,
            title="TOTAL RESPONSES PERCENTAGE REPRESENTATION",
            template="plotly_white",
        )

        # comments dataframe and manipulation
        comments_df = hp.get_comments_dataframe(
            data_frame=grouped_data if not show_general_stats else dataframe
        )

        text = grouped_data["COMMENT"].replace(r'^\s*$', np.nan, regex=True).dropna() if not show_general_stats else \
            dataframe["COMMENT"].replace(r'^\s*$', np.nan, regex=True).dropna()
        word_cloud = hp.get_common_word_cloud(text)

        # summarize the text
        all_comment_text = ""
        for txt in text:
            if txt is not None or txt != "" or len(txt) <= 0:
                all_comment_text += f" {txt}"
        summary = hp.summarize_comment(text=all_comment_text)
        # print(summary)

        text = grouped_data["COMMENT"] if not show_general_stats else dataframe["COMMENT"]
        comments_sentiment_dataframe = hp.get_comment_for_analysis(text)
        comment_sentiment_counts = hp.get_sentiment_stats(
            comments_sentiment_dataframe["Subjectivity"]
        )

        comments_sentiment_df = pd.DataFrame(
            sorted(comment_sentiment_counts.values(), reverse=True),
            columns=["COUNT"]
        )

        comments_bar_plot = px.bar(
            comments_sentiment_df,
            x=hp.sentiments_categories,
            y="COUNT",
            color="COUNT",
            title="TOTAL SENTIMENT REPRESENTATION",
            template="plotly_white"
        )

        comments_pie_chart = px.pie(
            comments_sentiment_df,
            values="COUNT",
            names=hp.sentiments_categories,
            title="TOTAL SENTIMENT PERCENTAGE REPRESENTATION",
            template="plotly_white",
        )

        # Display the generated image:
        word_cloud_figure, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(word_cloud)
        plt.axis("off")

        right_column, left_column = st.columns(2)
        right_column_sentiment, left_column_sentiment = st.columns(2)

        with left_column:
            st.subheader("TOTAL NUMBERS")
            st.dataframe(stats_df_panel)
            st.markdown('##')
            st.plotly_chart(stats_bar_plot)
        with right_column:
            st.subheader("TOTAL RESPONSES")
            st.plotly_chart(responses_bar_plot)
            st.plotly_chart(resp_pie_chart)

        with left_column_sentiment:
            st.plotly_chart(comments_pie_chart)
            st.subheader("COMMENT SUMMARY")
            st.markdown(summary)
        with right_column_sentiment:
            st.subheader("SENTIMENT ANALYSIS")
            st.plotly_chart(comments_bar_plot)
            st.subheader("COMMON WORDS CLOUD")
            st.pyplot(word_cloud_figure)

        # --- Download datasheets ----
        st.sidebar.subheader("DOWNLOADS")
        hp.generate_excel_download_link(grouped_data, "Data")
        hp.generate_excel_download_link(grouped_data, "Comments")
        hp.generate_html_download_link(stats_bar_plot, "Stats")
        hp.generate_html_download_link(resp_pie_chart, "Response pie")
        hp.generate_html_download_link(responses_bar_plot, "Response bar")
        hp.generate_html_download_link(comments_pie_chart, "Sentiment pie")
        hp.generate_html_download_link(comments_bar_plot, "Sentiment bar")

        if show_grouped_data:
            st.header("GROUPED DATA")
            st.markdown("Data grouped by your selection")
            st.dataframe(grouped_data)
            bottom_left, bottom_right = st.columns(2)
            with bottom_left:
                st.dataframe(comments_df)
            with bottom_right:
                st.dataframe(questions_dataframe)

        if show_raw_data:
            st.header("RAW DATA")
            st.markdown("The Raw Data Uploaded")
            st.dataframe(dataframe)
            bottom_left, bottom_right = st.columns(2)
            with bottom_left:
                st.dataframe(comments_df)
            with bottom_right:
                st.dataframe(questions_dataframe)
